# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math
from deepspeed.runtime.zero.partition_parameters import print_rank_0

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
    # Try to use FusedLayerNorm from Apex - this will trigger an error.
    _ = LayerNorm(8, eps=1e-5)

except Exception as e:
    print('WARNING: APEX is not installed, using torch.nn.LayerNorm '
          'instead of apex.normalization.FusedLayerNorm!')
    from torch.nn import LayerNorm

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args
import deepspeed.runtime.activation_checkpointing.checkpointing as ds_checkpointing

def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    
    if ds_checkpointing.is_configured():
        global get_cuda_rng_tracker
        get_cuda_rng_tracker = ds_checkpointing.get_cuda_rng_tracker
        
    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_model_parallel_rank(),
                self.model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weight
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        if not bias:
            self.skip_bias_add = True

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)
            
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        # XXX seq linear crash
        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel 
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)



    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

class SequentialParallelLinear(torch.nn.Module):
    """
        A wrapper around row / column parallel linear, which can be used to 
        break down a very large linear layer into a sequence of smaller ones.
        Useful when a single linear layer consumes too much working memory
        to fit in GPU.
        Arguments:
            input_size: first dimension of matrix A.
            output_size: second dimension of matrix A.
            bias: If true, add bias
            is_row_parallel: choses between RowLinearParallel vs ColumnLinearParallel
            row_splits: number of partitions along the row of the linear layer
            col_splits: number of partitions along the col of the linear layer
            input_is_already_split: input is already split to match the row_splits
            combine_col_splits: concatinates the outputs of the col splits
    """
    def __init__(self, input_size, output_size, bias=True,
                 is_row_parallel=True,
                 row_splits=1,
                 col_splits=1, 
                 input_is_already_split=False,
                 combine_col_splits=False,
                 **kwargs):

        super(SequentialParallelLinear, self).__init__()
        self.row_splits = row_splits
        self.col_splits = col_splits
        self.input_is_already_split = input_is_already_split
        self.combine_col_splits = combine_col_splits

        assert input_size % row_splits == 0, f"Cannot split input_size = {input_size} in row_splits = {row_splits}"
        assert output_size % col_splits == 0, f"Cannot split input_size = {input_size} in row_splits = {row_splits}"

        input_size_partition = input_size // row_splits
        output_size_partition = output_size // col_splits

        self.linears = torch.nn.ModuleList()
        #norm2=None
        for row_id in range(row_splits):
            self.linears.append(torch.nn.ModuleList())

            #if input_size is split, we only need one bias
            this_bias = bias if row_id == (row_splits-1) else False

            for col_id in range(col_splits):
                if is_row_parallel:
                    self.linears[row_id].append(RowParallelLinear(input_size_partition, output_size_partition, 
                                                                  bias=this_bias,
                                                                  **kwargs))
                else:
                    self.linears[row_id].append(ColumnParallelLinear(input_size_partition, output_size_partition, 
                                                                     bias=this_bias,
                                                                     **kwargs))

    def forward(self, input_):

        if self.row_splits > 1 and not self.input_is_already_split:
            inputs = split_tensor_along_last_dim(input_,self.row_splits)    
        elif self.row_splits > 1:
            inputs = input_
            assert len(inputs) == self.row_splits, f"Row splits {self.row_splits} does not match input splits {len(inputs)}"
        else:
            inputs = [input_]

        outputs=[]
        for row_id in range(self.row_splits):
            for col_id in range(self.col_splits):
                local_output = self.linears[row_id][col_id](inputs[row_id])
                if row_id == 0:
                    #this clone is necessary to preserve auto grad
                    #there is some issue with inplace update for outputs that are views
                    if torch.is_tensor(local_output):
                        outputs.append(local_output.clone())   
                    else:
                        outs = []
                        for out in local_output:
                            if torch.is_tensor(out):
                                outs.append(out.clone())
                            else:
                                outs.append(out)
                        outputs.append(outs)
                else:
                    if torch.is_tensor(local_output):
                        outputs[col_id] += local_output
                    else:
                        for idx, local_out in  enumerate(local_output):
                            try:
                                outputs[col_id][idx] += local_out
                            except TypeError:
                                # Can't accumulate, just assign
                                if not outputs[col_id][idx] == local_out:
                                    #print(f'RANK={torch.distributed.get_rank()} outputs={outputs[col_id][idx]} local={local_out}')
                                    pass
                                outputs[col_id][idx] = local_out


        if not self.combine_col_splits:
            ret = tuple(outputs)
        else:
            ret = []
            for idx in range(len(outputs[0])):
                if torch.is_tensor(outputs[0][idx]):
                    full_tensor = [out[idx] for out in outputs]
                    ret.append(torch.cat(full_tensor,dim=-1))
                else:
                    # not a tensor to cat, just grab the first one because they are
                    # all ensured to be the same above
                    ret.append(outputs[-1][idx])
        return ret

    @staticmethod
    def seq_parallel_splits(two_linear_config):
        """ two_linear_config is an 7 or 8 digit integer, where
        08040216 means
        lin0_col_splits is 8
        lin0_row_splits is 4
        lin1_col_splits is 2
        lin1_row splits is 16
        """

        lin0_cols = max((two_linear_config // 1000000) % 100,1)
        lin0_rows = max((two_linear_config // 10000) % 100,1)

        lin1_cols = max((two_linear_config // 100) % 100,1)
        lin1_rows = max(two_linear_config % 100, 1)

        return (lin0_cols, lin0_rows, lin1_cols, lin1_rows)

"""
#concatinates a list of tensors to produce a smaller list of output_count
#concatinates along dim
def concat_tensors(input_list,output_count=1, dim=-1):
    num_inputs = len(input_list)

    assert num_inputs % output_count == 0, f" Cannot concat len{input_list} to {output_count} outputs"
    num_inputs_per_output = num_inputs // output_count

    outputs = []
    for i in range(output_count):
        start = i * num_inputs_per_output
        end = start + num_inputs_per_output
        outputs.append(torch.cat(input_list[start:end],dim=dim))

    return outputs
"""