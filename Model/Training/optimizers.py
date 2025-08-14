import torch


def get_sgd_optim(model: torch.nn.Module, base_lr: float = 0.1, momentum: float = 0.9,
                  nesterov: bool = True, weight_decay: float = 0.00004) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=base_lr, momentum=momentum,
                                weight_decay=weight_decay, nesterov=nesterov)
    return optimizer


def get_adam_optim(model: torch.nn.Module, base_lr: float = 0.001, betas: float = (0.9, 0.999),
                   weight_decay: float = 0.00004) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr,
                                 betas=betas, weight_decay=weight_decay, eps=0.00001)
    return optimizer


def get_adamw_optim(model: torch.nn.Module, base_lr: float = 0.001,
                    betas: float = (0.9, 0.999),
                    weight_decay: float = 0.00004) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
                                  model.parameters()), lr=base_lr,
                                  betas=betas, weight_decay=weight_decay,
                                  eps=0.00001)
    return optimizer


def get_current_lr(optim):
    num_param_groups = optim.param_groups
    if len(num_param_groups) == 1:
        return optim.param_groups[0]['lr']
    else:
        lrs = []
        for i in range(num_param_groups):
            lrs.append(optim.param_groups[i]['lr'])
        return lrs
