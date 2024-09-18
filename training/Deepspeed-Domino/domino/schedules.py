# Copied and modified from Megatron-LM

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import domino.parallel_state as parallel_state
from domino.modules.enums import ModelType
from domino.utils import get_attr_wrapped_model, get_model_config, get_model_type
from domino.utils import average_losses_across_data_parallel_group

from megatron import core
from megatron.core.pipeline_parallel import p2p_communication

def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    """
    forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    if parallel_state.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss = output_tensor
            averaged_loss = average_losses_across_data_parallel_group([loss])
            loss_reduced = {"lm loss": averaged_loss[0]}
            output_tensor = loss / num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    # model_type = get_model_type(model)
    # if (
    #     parallel_state.is_pipeline_stage_after_split()
    #     and model_type == ModelType.encoder_and_decoder
    # ):
    #     return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    no_sync_func = config.no_sync_func
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
            )
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
    end.record()
    torch.cuda.synchronize()

    if is_rank_0():
        print('backward time: ')
        print(start.elapsed_time(end))
    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors
