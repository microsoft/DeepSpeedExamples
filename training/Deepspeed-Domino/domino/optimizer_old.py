from apex.optimizers import FusedAdam as Adam

from domino.arguments import get_args 

from deepspeed.runtime.fp16.fused_optimizer import FP16_Optimizer
from deepspeed.utils.timer import SynchronizedWallClockTimer


def get_param_groups(modules,
                     no_weight_decay_cond,
                     scale_lr_cond):
    wd_no_scale_lr = []
    no_wd_no_scale_lr = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            if no_weight_decay_cond is not None:
                no_wd = no_weight_decay_cond(name, param)
            else:
                # do not regularize biases nor Norm parameters
                no_wd = name.endswith(".bias") or len(param.shape) == 1

            if scale_lr_cond is not None:
                scale_lr = scale_lr_cond(name, param)
            else:
                scale_lr = False

            if not no_wd and not scale_lr:
                wd_no_scale_lr.append(param)
            elif no_wd and not scale_lr:
                no_wd_no_scale_lr.append(param)

    param_groups = []
    if len(wd_no_scale_lr):
        param_groups.append({'params': wd_no_scale_lr, 'wd_mult': 1.0, 'lr_mult': 1.0})
    if len(no_wd_no_scale_lr):
        param_groups.append({'params': no_wd_no_scale_lr, 'wd_mult': 0.0, 'lr_mult': 1.0})

    return param_groups

def get_domino_optimizer(model,
                         no_weight_decay_cond=None,
                         scale_lr_cond=None):
    args = get_args()

    # Base optimizer.
    model_parameters = get_param_groups(model,
                                    no_weight_decay_cond,
                                    scale_lr_cond)

    if args.optimizer == 'adam':
        optimizer = Adam(model_parameters,
                         lr=args.lr,
                         weight_decay=args.weight_decay,
                         betas=(args.adam_beta1, args.adam_beta2),
                         eps=args.adam_eps)
    else:
        raise Exception('{} optimizer is not supported.'.format(
            args.optimizer))

    return FP16_Optimizer(init_optimizer=optimizer,
                          deepspeed=None,
                          static_loss_scale=1.0,
                          dynamic_loss_scale=True,
                          initial_dynamic_scale=131072,
                          dynamic_loss_args=None,
                          verbose=False,
                          mpu=None,
                          clip_grad=args.clip_grad,
                          fused_adam_legacy=False,
                          has_moe_layers=False,
                          timers=SynchronizedWallClockTimer())
