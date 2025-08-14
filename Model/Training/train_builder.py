import os

import numpy as np

import Training.optimizers as optimizers
import Training.lr_schedulers as lr_schedulers
import Training.loss_functions as loss_functions
import Dataloaders.human_seg_dl as human_seg_dl
import data_fps


def get_dataset(cfg: dict, dl_kwargs: dict, train_d: dict) -> None:
    overall_dataset_name = cfg["overall_dataset_name"]
    if overall_dataset_name == "Human_3d_Seg":
        dataset_name = cfg.get("dataset_name")
        with_random_rot = cfg.get("with_random_rot", True)
        with_point_jitter = cfg.get("with_point_jitter", True)
        preprocess_func_name = cfg.get("preprocess_func_name", None)
        pre_process_func = human_seg_dl.get_preprocess_func(preprocess_func_name)
        if pre_process_func is not None:
            train_d["preprocess_func"] = pre_process_func
        is_sapiens_v2 = cfg.get("is_sapiens_v2", False)
        dataset_kwargs = {"dataset_name": dataset_name, "with_random_rot": with_random_rot,
                          "with_point_jitter": with_point_jitter, "is_sapiens_v2": is_sapiens_v2}
        train_dataset_kwargs = dataset_kwargs.copy()
        train_dataset_kwargs["split"] = "trainval"
        test_dataset_kwargs = dataset_kwargs.copy()
        test_dataset_kwargs["split"] = "test"

        train_dl_kwargs = dl_kwargs.copy()
        train_dl_kwargs["shuffle"] = True
        test_dl_kwargs = dl_kwargs.copy()
        test_dl_kwargs["shuffle"] = False
        test_dl_kwargs["drop_last"] = False

        train_dl = human_seg_dl.get_human3d_dataloader(train_dataset_kwargs, train_dl_kwargs)

        test_dl = human_seg_dl.get_human3d_dataloader(test_dataset_kwargs, test_dl_kwargs)
        train_d["train_dl"] = train_dl
        train_d["test_dl"] = test_dl

        # Paths for evaluation
        train_d["sampled_dataset_vertices_fp"] = human_seg_dl.get_hdf5_file_fp(dataset_name, is_sapiens_v2)
        train_d["original_dataset_vertices_fp"] = os.path.join(data_fps.data_fps["Renders_Outer"],
                                                               "PCs", "reoriented_vertices.hdf5")
        train_d["original_dataset_labels_fp"] = human_seg_dl.get_original_labels_fp(dataset_name, train_dataset_kwargs["is_sapiens_v2"])

        if "sapiens" in dataset_name:
            if is_sapiens_v2:
                train_d["is_sapiens_v2"] = True
                train_d["cm_num_to_str"] = human_seg_dl.sapiens_v2_cm_num_to_str
                train_d["cm_str_to_num"] = human_seg_dl.sapiens_v2_cm_str_to_num
            else:
                train_d["is_sapiens_v2"] = False
                train_d["cm_num_to_str"] = human_seg_dl.sapiens_v1_cm_num_to_str
                train_d["cm_str_to_num"] = human_seg_dl.sapiens_v1_cm_str_to_num
        else:
            train_d["cm_num_to_str"] = human_seg_dl.cihp_cm_num_to_str
            train_d["cm_str_to_num"] = human_seg_dl.cihp_cm_str_to_num

    else:
        raise ValueError(f"Dataset {overall_dataset_name} not supported.")


def get_optimizer(cfg: dict, train_d: dict) -> None:
    optimizer_name = cfg["optimizer_type"]
    if optimizer_name == "sgd":
        base_lr = cfg["base_lr"]
        weight_decay = cfg.get("weight_decay", 0.00005)
        nesterov = cfg.get("nesterov", True)
        momentum = cfg.get("momentum", 0.90)
        optim = optimizers.get_sgd_optim(model=train_d["model"], base_lr=base_lr, weight_decay=weight_decay,
                                         nesterov=nesterov, momentum=momentum)
    elif optimizer_name == "adamw":
        base_lr = cfg["base_lr"]
        weight_decay = cfg.get("weight_decay", 0.00005)
        optim = optimizers.get_adamw_optim(train_d["model"], base_lr=base_lr, weight_decay=weight_decay)
    else:
        print(f"Optimizer {optimizer_name} is not supported.")
        exit(-1)
    train_d["optimizer"] = optim


def get_lrs(cfg: dict, train_d: dict) -> None:
    lrs_name = cfg["lrs_scheduler"]
    if lrs_name == "ms_lrs":
        gamma_val = cfg["gamma_val"]
        time_interval = cfg["time_interval"]
        num_epochs = cfg["num_epochs"]
        lr_scheduler = lr_schedulers.get_k_step_lrs(optimizer=train_d["optimizer"], num_epochs=num_epochs,
                                                    gamma_val=gamma_val, interval=time_interval)
    elif lrs_name == "cosine_annealing":
        num_epochs = cfg["num_epochs"]
        min_learning_rate = cfg.get("min_learning_rate", 0.000001)
        lr_scheduler = lr_schedulers.get_cos_annealing_lrs(optimizer=train_d["optimizer"], num_epochs=num_epochs,
                                                           min_learning_rate=min_learning_rate)
    elif lrs_name == "evenly_distributed_lr":
        num_epochs = cfg["num_epochs"]
        min_learning_rate = cfg.get("min_learning_rate", 0.000001)
        starting_lr = cfg["base_lr"]
        lr_scheduler = lr_schedulers.get_evenly_distributed_lr(optimizer=train_d["optimizer"], starting_lr=starting_lr,
                                                               num_epochs=num_epochs, min_learning_rate=min_learning_rate)
    else:
        raise ValueError(f"Learning rate scheduler {lrs_name} not supported.")

    train_d["lr_scheduler"] = lr_scheduler


def get_loss_function(cfg: dict, train_d: dict) -> None:
    lf_name = cfg["loss_fn"]
    if lf_name == "cross_entropy":
        label_smoothing = cfg.get("label_smoothing", 0.0)
        ignore_index = cfg.get("ignore_index", -100)
        with_ce_weights = cfg.get("with_ce_weights", False)
        if with_ce_weights:
            ce_weights = train_d["class_weights"]
        else:
            ce_weights = None
        loss_fn = loss_functions.get_ce_loss(label_smoothing, ignore_index, ce_weights)
    elif lf_name == "cross_entropy_and_dice":
        ce_loss_weight = cfg.get("ce_loss_weight", 0.5)
        dice_loss_weight = cfg.get("dice_loss_weight", 0.5)
        loss_fn = loss_functions.CEDiceLoss(ce_weight=ce_loss_weight, dice_weight=dice_loss_weight)
    else:
        raise ValueError(f"Loss function {lf_name} not supported.")
    train_d["loss_fn"] = loss_fn


def train_build_from_cfg(cfg: dict, train_d: dict) -> None:
    # dataloader arguments
    batch_size = cfg["batch_size"]
    train_d["with_ddp"] = cfg.get("with_ddp", False)
    num_workers = cfg.get("num_workers", 0)

    # At the moment persistent workers with DDP seems to be causing indefinite hanging, #TODO investigate this further
    persistent_workers = num_workers != 0 and not train_d["with_ddp"]
    pin_memory = cfg.get("pin_memory", False)
    dl_kwargs = {"batch_size": batch_size, "num_workers": num_workers,
                 "persistent_workers": persistent_workers, "pin_memory": pin_memory}
    train_d["batch_size"] = batch_size

    train_d["set_device"] = cfg.get("set_device", False)
    train_d["num_epochs"] = cfg["num_epochs"]
    train_d["with_testing"] = cfg.get("with_testing", True)
    train_d["with_train_accuracy"] = cfg.get("with_train_accuracy", False)
    train_d["starting_test_epoch"] = cfg.get("starting_test_epoch", train_d["num_epochs"] // 2)
    train_d["with_16_bit_precision"] = cfg.get("with_16_bit_precision", False)
    train_d["with_grad_clip"] = cfg.get("with_grad_clip", False)
    train_d["grad_clip_val"] = cfg.get("grad_clip_val", None)

    # dataset
    get_dataset(cfg, dl_kwargs, train_d)

    # optimizer and lr scheduler
    get_optimizer(cfg, train_d)
    get_lrs(cfg, train_d)

    # loss function
    get_loss_function(cfg, train_d)
