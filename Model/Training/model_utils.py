import torch


def set_model_all_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def load_model_from_save(model_save_path: str, model: torch.nn.Module) -> torch.nn.Module:
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_sd'])
    return model


def save_model_during_training(epoch, model, optimizer, lr_scheduler, logs_dict, save_path):
    torch.save({
            'epoch_num': epoch,
            'model_sd': model.state_dict(),
            'optimizer_sd': optimizer.state_dict(),
            'lrs_sd': lr_scheduler.state_dict() if lr_scheduler is not None else -1,
            'logs_dict': logs_dict
    }, save_path)


def save_model_only_during_training(model, save_path):
    torch.save({
            'model_sd': model.state_dict(),
    }, save_path)


# DDP adds the word Module. to all the module names upon saving
def convert_ddp_sd(sd):
    new_sd = {}
    for key in sd.keys():
        new_sd[key[7:]] = sd[key]
    return new_sd


def load_from_ddp(model_save_path, model):
    checkpoint = torch.load(model_save_path)
    saved_sd = checkpoint["model_sd"]
    converted_sd = convert_ddp_sd(saved_sd)
    model.load_state_dict(converted_sd)
    return model