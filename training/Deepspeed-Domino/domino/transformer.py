from contextlib import nullcontext
import numpy as np
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from domino.arguments import get_args, get_num_microbatches
from domino.modules.module import DominoModule
import domino.parallel_state as mpu
from domino.modules.fused_layer_norm import MixedFusedLayerNorm
from domino.modules.fused_layer_norm import MixedFusedLayerNorm as LayerNorm
from domino.modules.enums import AttnMaskType, LayerType, AttnType, ModelType
from domino.tensor_parallel.partition import _initialize_affine_weight_gpu, set_tensor_model_parallel_attributes
from domino.tensor_parallel.partition import ColumnParallelLinear, RowParallelLinearNoComm

from domino.utils import make_viewless_tensor

def is_rank_0():
    if torch.distributed.get_rank() == 0:
        return True

class AddDropoutFuseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        if bias1 is not None and bias2 is not None:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_add_fuse(
                input1, bias1, residual1, input2, bias2, residual2, prob, training
            )
        else:
            output1, mask1, output2, mask2 = torch._C._nn.native_add_dropout_fuse(
                input1, residual1, input2, residual2, prob, training
            )
        scale = 1.0 / (1.0 - prob)
        ctx.save_for_backward(mask1, mask2)
        ctx.scale = scale
        ctx.with_bias = bias1 is not None and bias2 is not None
        return output1, output2

    @staticmethod
    def backward(ctx, grad_output1, grad_output2):
        (mask1, mask2) = ctx.saved_tensors
        scale = ctx.scale
        with_bias = ctx.with_bias
        if with_bias:
            grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2 = (
                torch._C._nn.native_add_dropout_add_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
        else:
            grad_input1, grad_residual1, grad_input2, grad_residual2 = (
                torch._C._nn.native_add_dropout_fuse_2(grad_output1, mask1, grad_output2, mask2, scale)
            )
            grad_bias1 = None
            grad_bias2 = None
        return grad_input1, grad_bias1, grad_residual1, grad_input2, grad_bias2, grad_residual2, None, None


class AddDropoutFuse(torch.nn.Module):
    def __init__(self):
        super(AddDropoutFuse, self).__init__()

    def forward(self, input1, bias1, residual1, input2, bias2, residual2, prob, training):
        return AddDropoutFuseFunction.apply(input1, bias1, residual1, input2, bias2, residual2, prob, training)

def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)



def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)

handle_dic = {}

def no_oper(input_, dic_, h_id):
    return NoOper.apply(input_, dic_, h_id)

class NoOper(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, handle_dic, h_id):
        return input_

    @staticmethod
    def forward(ctx, input_, handle_dic, h_id):
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # if is_rank_0():
        #     print(f"===Guanhua ctx handle {ctx.handle_dic}")
        handle = ctx.handle_dic[ctx.h_id]
        handle.wait()
        return grad_output, None, None

def copy_to_tensor_model_parallel_region_a(input_, dic_, h_id):
    return _CopyToModelParallelRegionA.apply(input_, dic_, h_id)

class _CopyToModelParallelRegionA(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, handle_dic, h_id):
        return input_

    @staticmethod
    def forward(ctx, input_, handle_dic, h_id):
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # Bypass the function if we are using only 1 GPU.
        if mpu.get_tensor_model_parallel_world_size() == 1:
            return grad_output

        # Async All-reduce.
        handle = torch.distributed.all_reduce(grad_output, group=mpu.get_tensor_model_parallel_group(), async_op=True)
        ctx.handle_dic[ctx.h_id] = handle
        return grad_output, None, None

class DropPath(DominoModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class CoreAttention(DominoModule):

    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()

        self.layer_number = max(1, layer_number)
        self.att_dropout_p = config.attention_dropout
        self.is_causal = True
        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = projection_size // world_size

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        # [sq, b, np, hn] -> [b, np, sq, hn]
        query_layer = query_layer.permute(1, 2, 0, 3).contiguous()

        # [sk, b, np, hn] -> [b, np, sk, hn]
        key_layer = key_layer.permute(1, 2, 0, 3).contiguous()

        # [sk, b, np, hn] -> [b, np, sk, hn]
        value_layer = value_layer.permute(1, 2, 0, 3).contiguous()

        # set attn_mask as None to match is_causal=True
        context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attn_mask=None, 
                                                                dropout_p=self.att_dropout_p, is_causal=True, scale=1.0)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SharedAttention(DominoModule):
    """Shared self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(SharedAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype

        self.group_query_attention = args.group_query_attention
        self.num_query_groups = args.num_query_groups

        query_projection_size = config.kv_channels * config.num_attention_heads
        if self.group_query_attention:
            kv_projection_size = args.kv_channels * args.num_query_groups
        else:
            kv_projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = query_projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads // world_size

        if self.group_query_attention:
            if args.num_query_groups % world_size != 0:
                raise NotImplementedError('Currently the num_query_groups should be '
                                          'a multiple of the tensor parallel size')
            self.num_query_groups_per_partition = args.num_query_groups // world_size
        else:
            self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        self.query_key_value = ColumnParallelLinear(
            config.hidden_size,
            query_projection_size + 2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=args.add_bias_linear,
            gather_output=False)

        self.core_attention = CoreAttention(self.layer_number, config,
                                            self.attn_mask_type)

        # Output.
        self.dense = RowParallelLinearNoComm(
            query_projection_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True)

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        encoder_output=None
        rotary_pos_emb=None

        # =====================
        # Query, Key, and Value
        # =====================
        # if self.attention_type == AttnType.self_attn:
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
        (query_layer,
        key_layer,
        value_layer) = torch.split(
            mixed_x_layer,
            [
                (
                    self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head
            ],
            dim=3)
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)

        # ==================================
        # core attention computation
        # ==================================
        repeat = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        assert repeat == 1, '[csy] disable the query groups to enable cuda graph'
        # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
        # key_layer = key_layer.repeat_interleave(
        #     self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        #     dim = 2
        # )
        # value_layer = value_layer.repeat_interleave(
        #     self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        #     dim = 2
        # )

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        context_layer = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask)
    
        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


# stream1 = torch.cuda.Stream()

class ParallelTransformerLayer(DominoModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
                 # retriever=None):
        args = get_args()
        self.args = args
        self.args.llama_model = False

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = MixedFusedLayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm)

        # Self attention.
        self.self_attention_1 = SharedAttention(
            config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)

        self.self_attention = self.self_attention_1

        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = MixedFusedLayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm)

        # Cross attention.
        if self.layer_type in (LayerType.decoder,):
            self.inter_attention = SharedAttention(
                config,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = MixedFusedLayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=not config.persist_layer_norm)

        # MLP
        # ------------ explicit mlp start ------------
        ffn_hidden_size = config.ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
    
        input_size = config.hidden_size
        output_size = ffn_hidden_size
        init_method=config.init_method
        output_layer_init_method=config.output_layer_init_method
        bias=config.add_bias_linear
        gather_output=False
        stride=1
        keep_master_weight_for_test=False
        skip_bias_add=True
        skip_weight_param_allocation = False
        is_expert=False

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.output_size_per_partition = output_size // world_size
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        # self.expert_parallel = config.expert_model_parallel_size > 1
        self.expert_parallel = False
        self.config = config

        # Keep input parameters
        self.input_size_r = config.ffn_hidden_size
        self.output_size_r = input_size
        # self.input_is_parallel = input_is_parallel
        self.input_is_parallel = False

        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = self.input_size_r // world_size

        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if not skip_weight_param_allocation:   
            self.weight_c = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            self.weight_r = Parameter(
                torch.empty(
                    self.output_size_r,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )

            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight_c,
                    init_method,
                    partition_dim=0,
                    stride=stride,
                    # expert_parallel=(self.is_expert and self.expert_parallel),
                )

                _initialize_affine_weight_gpu(
                    self.weight_r,
                    output_layer_init_method,
                    partition_dim=1,
                    stride=stride,
                    # expert_parallel=(self.is_expert and self.expert_parallel),
                )
            setattr(self.weight_c, 'allreduce', not (self.is_expert and self.expert_parallel))
            setattr(self.weight_r, 'allreduce', not (self.is_expert and self.expert_parallel))
        else:
            self.weight_c = None

        if bias:
            if config.use_cpu_initialization:
                self.bias_c = Parameter(
                    torch.empty(self.output_size_per_partition, dtype=config.params_dtype)
                )
                self.bias_r = Parameter(
                    torch.empty(self.output_size_r, dtype=config.params_dtype)
                )
            else:
                self.bias_c = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
                self.bias_r = Parameter(
                    torch.empty(
                        self.output_size_r,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias_c, True, 0, stride)
            set_tensor_model_parallel_attributes(self.bias_r, True, 0, stride)
            if config.perform_initialization:
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias_c.zero_()
                    self.bias_r.zero_()
        else:
            self.register_parameter('bias_c', None)
            self.register_parameter('bias_r', None)

        if args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.mlp_activation_func = swiglu
        else:
            self.mlp_activation_func = F.gelu

        self.async_tensor_model_parallel_allreduce = (
            config.async_tensor_model_parallel_allreduce and world_size > 1
        )

        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel and world_size <= 1:
            warnings.warn(
                f"`sequence_parallel` is set to `True`, but tensor model parallel size is {world_size}. "
                f"Disabling sequence parallel."
            )
            self.sequence_parallel = False

        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel` "
                "cannot be enabled at the same time."
            )

        self.explicit_expert_comm = self.is_expert and (
            self.sequence_parallel or self.expert_parallel
        )
        # ------------ explicit mlp end ------------

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        # if args.retro_add_retriever:
        #     retro_args = get_retro_args()
        #     self.retro_num_neighbors = args.retro_num_neighbors
        #     self.retro_chunk_length = retro_args.retro_gpt_chunk_length
        #     self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        # if layer_type == LayerType.retro_decoder_with_retriever:
        #     self.retriever = ParallelTransformer(
        #         config=config,
        #         model_type=ModelType.retro_encoder,
        #         self_attn_mask_type=AttnMaskType.padding,
        #         pre_process=True,
        #         post_process=False,
        #     )
        #     self._retriever_key = 'retriever'
        # else:
        #     self.retriever = None


    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [s, b, h]

        hidden_states0, hidden_states1 = hidden_states
        # Layer norm at the beginning of the transformer layer.
        layernorm_output0 = self.input_layernorm(hidden_states0)
        # import pdb; pdb.set_trace()
        layernorm_output1 = self.input_layernorm(hidden_states1)

        # self attention with cuda graph.
        if not self.args.llama_model:
            attention_output0, attention_bias0 = \
                self.self_attention(
                    layernorm_output0,
                    attention_mask)
        
        # if is_rank_0():
        #     print(f"===DEBUG: layernorm_output0 shape {layernorm_output0.shape} dtype {layernorm_output0.dtype}, attn_mask shape is {attention_mask.shape} dtype {attention_mask.dtype}, infer_params {inference_params}, rotary_pos_emb {rotary_pos_emb}\n")

        # attention_bias0= None
        # attention_bias1 = None
        # Self attention.
        else:
            attention_output0, attention_bias0  = \
                self.self_attention(
                    layernorm_output0,
                    attention_mask, #rotary_pos_emb)
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)
        # handle0 = torch.distributed.all_reduce(attention_output0, group=mpu.get_tensor_model_parallel_group())
        handle0 = torch.distributed.all_reduce(attention_output0, group=mpu.get_tensor_model_parallel_group(), async_op=True)

        # cuda graph
        if not self.args.llama_model:
            attention_output1, attention_bias1 = \
                self.self_attention(
                    layernorm_output1,
                    attention_mask)
        else:
            attention_output1, attention_bias1 = \
                self.self_attention(
                layernorm_output1,
                attention_mask,# rotary_pos_emb)
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb)
        handle1 = torch.distributed.all_reduce(attention_output1, group=mpu.get_tensor_model_parallel_group(), async_op=True)
        handle0.wait()

        # Residual0 connection.
        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
        else:
            residual0 = hidden_states0

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)
            
            if attention_bias0 is not None:
                attention_bias0 = attention_bias0.expand_as(residual0)
            with self.bias_dropout_add_exec_handler():
                layernorm_input0 = bias_dropout_add_func(
                    attention_output0,
                    attention_bias0,
                    residual0,
                    self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output0 = self.post_attention_layernorm(layernorm_input0)
        layernorm_output0 = no_oper(layernorm_output0, handle_dic, f'{self.layer_number}_0')

        # Residual1 connection.
        if self.apply_residual_connection_post_layernorm:
            residual1 = layernorm_output1
        else:
            residual1 = hidden_states1

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias1 is not None:
                attention_bias1 = attention_bias1.expand_as(residual1)
            with self.bias_dropout_add_exec_handler():
                layernorm_input1 = bias_dropout_add_func(
                    attention_output1,
                    attention_bias1,
                    residual1,
                    self.hidden_dropout)

        layernorm_output1 = self.post_attention_layernorm(layernorm_input1)
        layernorm_output1 = no_oper(layernorm_output1, handle_dic, f'{self.layer_number}_1')

        # ------------ explicit mlp start ------------

        bias_c = self.bias_c if not self.skip_bias_add else None

        input0 = copy_to_tensor_model_parallel_region_a(layernorm_output0, handle_dic, f'{self.layer_number}_0')

        # Batch0 Matrix multiply.
        output0 = torch.matmul(input0, self.weight_c.t())
        if bias_c is not None:
            output0 = output0 + bias_c
        output0 = self.mlp_activation_func(output0)
        output0 = torch.matmul(output0, self.weight_r.t())
        handle2 = torch.distributed.all_reduce(output0, group=mpu.get_tensor_model_parallel_group(), async_op=True)

        handle1.wait()
        # Batch1 Matrix multiply.
        # if self.async_tensor_model_parallel_allreduce or self.sequence_parallel:
        #     input1 = layernorm_output1
        # else:
            # input1 = copy_to_tensor_model_parallel_region(layernorm_output1)
        input1 = copy_to_tensor_model_parallel_region_a(layernorm_output1, handle_dic, f'{self.layer_number}_1')

        # stream1 = get_stream(1)
        # with torch.cuda.stream(stream1):
        output1 = torch.matmul(input1, self.weight_c.t())
        output1 = self.mlp_activation_func(output1)
        if bias_c is not None:
            output1 = output1 + bias_c
        output1 = torch.matmul(output1, self.weight_r.t())
        torch.distributed.all_reduce(output1, group=mpu.get_tensor_model_parallel_group()) 

        handle2.wait()

        output0 = output0 + self.bias_r if self.bias_r is not None else output0
        output1 = output1 + self.bias_r if self.bias_r is not None else output1
        output_bias = None

        mlp_output0, mlp_output1, mlp_bias0, mlp_bias1 = output0, output1, output_bias, output_bias
        # ------------ explicit mlp end ------------

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual0 = layernorm_output0
            residual1 = layernorm_output1
        else:
            residual0 = layernorm_input0
            residual1 = layernorm_input1

        if self.drop_path is None:
            if mlp_bias0 is not None:
                mlp_bias0 = mlp_bias0.expand_as(residual0)
                mlp_bias1 = mlp_bias1.expand_as(residual1)
            with self.bias_dropout_add_exec_handler():
                output0 = bias_dropout_add_func(
                    mlp_output0,
                    mlp_bias0,
                    residual0,
                    self.hidden_dropout)
                output1 = bias_dropout_add_func(
                    mlp_output1,
                    mlp_bias1,
                    residual1,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output0 = make_viewless_tensor(inp = output0,
                                                     requires_grad = output0.requires_grad,
                                                     keep_graph = True)
            output1 = make_viewless_tensor(inp = output1,
                                                     requires_grad = output1.requires_grad,
                                                     keep_graph = True)
        else:
            if mlp_bias0 is not None:
                mlp_output0 = mlp_output0 + mlp_bias0
                mlp_output1 = mlp_output1 + mlp_bias1                
            out0 = torch.nn.functional.dropout(mlp_output0,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output0 = residual0 + self.drop_path(out0)
            out1 = torch.nn.functional.dropout(mlp_output1,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output1 = residual1 + self.drop_path(out1)

        return output0, output1


def _get_num_layers(args, model_type, is_decoder=False):
    if not is_decoder:
        num_layers = args.encoder_num_layers
    else:
        num_layers = args.decoder_num_layers
    return num_layers


class ParallelTransformer(DominoModule):
    """Transformer class."""
    def __init__(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()
        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = config.bf16
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.num_layers = _get_num_layers(args, model_type, layer_type==LayerType.decoder)
        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, config.num_layers)]

        def build_layer(layer_number):
            return ParallelTransformerLayer(
                config,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        # offset = self.num_layers
        offset = 0
        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])
    
        if self.post_process and self.post_layer_norm:
            self.final_layernorm = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input."""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [s, b, h]
        if not self.pre_process:
            hidden_states = self.input_tensor

        hidden_states0, hidden_states1 = hidden_states
        hidden_states0 = make_viewless_tensor(
            hidden_states0,
            requires_grad=True,
            keep_graph=True,
        )
        hidden_states1 = make_viewless_tensor(
            hidden_states1,
            requires_grad=True,
            keep_graph=True,
        )
        hidden_states = [hidden_states0, hidden_states1]

        # # Determine if the current iteration is first microbatch
        # if self.num_microbatches_in_previous_step != get_num_microbatches():
        #     self.microbatch_count = 0 # Reset count on new batch size rampup interval
        # self.num_microbatches_in_previous_step = get_num_microbatches()
        # is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

        forward_kwargs = {
            'encoder_output': encoder_output,
            'enc_dec_attn_mask': enc_dec_attn_mask,
            'inference_params': inference_params,
        }
        forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

        for index in range(self.num_layers):
            # set_ite_layer(None, n_l=index)
            layer = self._get_layer(index)

            hidden_states = layer(
                hidden_states,
                attention_mask,
                **forward_kwargs)

        # Skip counter update for eval and activation checkpointing
        if torch.is_grad_enabled() and self.training:
            self.microbatch_count += 1

        hidden_states0, hidden_states1 = hidden_states
        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states0 = self.final_layernorm(hidden_states0)
            hidden_states1 = self.final_layernorm(hidden_states1)

        return hidden_states0, hidden_states1