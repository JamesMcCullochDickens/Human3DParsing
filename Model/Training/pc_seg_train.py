from typing import *
import os
import random
import time
import gc

import tqdm
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

import utils.plot_utils as plot_utils
import Training.dist_utils as dist_utils
import Training.optimizers as optims
import Training.model_utils as model_utils
import Evaluation.pc_seg_eval as part_seg_eval


def train_launch(train_d: dict) -> None:
    num_gpus = torch.cuda.device_count()
    if train_d["with_ddp"] and num_gpus > 1:
        if "gpu_override" not in train_d.keys():
            train_d["gpu_override"] = list(range(num_gpus))
        world_size = len(train_d["gpu_override"])
        print("Spawning processes")
        dist_utils.spawn_processes(fn=part_seg_train, args=(train_d, world_size))
    else:
        if "gpu_override" in train_d.keys():
            rank = train_d["gpu_override"][0]
        else:
            rank = 0
        part_seg_train(rank=rank, train_d=train_d, world_size=1)


def init_seed(seed: float = 1.0) -> None:
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(int(seed))
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def inp_to_gpu(inp, gpu_num: int):
    if isinstance(inp, (torch.Tensor)):
        return inp.to(gpu_num)
    elif isinstance(inp, List):
        for i in range(len(inp)):
            if isinstance(inp[i], torch.Tensor):
                inp[i] = inp[i].to(gpu_num)
    else:
        raise ValueError(f"Type {type(int)} not supported for inp_to_gpu")


def part_seg_train(rank: int, train_d: dict, world_size: int) -> None:
    if world_size > 1:
        dist_utils.setup(rank, world_size)
        gpu_num = train_d["gpu_override"][rank] # reset in case of change of default gpu order
        cpu_core_count = os.cpu_count()
    else:
        gpu_num = rank
    init_seed()

    # save fps
    model_name = train_d["model_name"]
    save_fps = train_d["save_fps"]
    model_save_fp = os.path.join(save_fps["trained_models_fp"], model_name)
    logs_save_fp = save_fps["training_logs_fp"]

    # optimizer and lrs_scheduler
    optimizer = train_d["optimizer"]
    lrs = train_d["lr_scheduler"]

    # training/eval stats
    epoch_overall_losses = []
    epoch_average_losses = []
    iteration_losses = []
    test_miou_vals = []
    test_epochs = []
    best_test_miou = 0.0

    model = train_d["model"]
    model.num_eval_clips = 1
    model.to(gpu_num)
    model.train()

    # loading from a previous training session
    if train_d["with_load"]:
        if not os.path.exists(model_save_fp):
            print("No file to load from. Will now start a new .pt save.\n")
            epoch_num_ = 1
        else:
            checkpoint = torch.load(model_save_fp)
            model.load_state_dict(checkpoint["model_sd"])
            optimizer.load_state_dict(checkpoint["optimizer_sd"])
            lrs.load_state_dict(checkpoint["lrs_sd"])
            epoch_num_ = checkpoint["epoch_num"]

            # epoch training logs
            if "logs_dict" in checkpoint.keys() and rank == 0:
                epoch_overall_losses = checkpoint["logs_dict"]["epoch_overall_losses"]
                epoch_average_losses = checkpoint["logs_dict"]["epoch_average_losses"]
                iteration_losses = checkpoint["logs_dict"]["iteration_losses"]
                test_miou_vals = checkpoint["logs_dict"]["test_miou_vals"]
                test_epochs = checkpoint["logs_dict"]["test_epochs"]
                best_test_miou = checkpoint["logs_dict"]["best_test_accuracy"]
    else:
        epoch_num_ = 1

    # loss function
    loss_fn = train_d["loss_fn"]

    # dataloaders train and test
    train_dataloader = train_d["train_dl"]
    test_dataloader = train_d.get("test_dl", None)

    # optional preprocess function
    preprocess_func = train_d.get("preprocess_func", None)

    # ddp dataloader declaration and model modifications
    if world_size > 1:
        # sharded dataloaders
        if rank == 0:
            print("Getting distributed dataloaders and DPP model...")
        train_sampler = DistributedSampler(train_dataloader.dataset, num_replicas=world_size,
                                           rank=rank, shuffle=True, drop_last=True)
        if "num_workers" not in train_d.keys():
            num_workers = cpu_core_count // world_size
        else:
            if train_d["num_workers"] == 1:
                num_workers = 1
            else:
                num_workers = train_d["num_workers"] // world_size
        train_dataset = train_dataloader.dataset
        train_dataloader = DataLoader(sampler=train_sampler, dataset=train_dataset,
                                      batch_size=train_d["batch_size"] // world_size,
                                      collate_fn=train_dataloader.collate_fn,
                                      num_workers=num_workers,
                                      drop_last=True,
                                      persistent_workers=num_workers > 0)
        # ddp
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu_num], output_device=gpu_num,
                    find_unused_parameters=False)
        if rank == 0:
            print("Finished getting distributed dataloaders and DDP model.")

    # training parameters
    num_epochs = train_d["num_epochs"]
    with_train_accuracy = train_d.get("with_train_accuracy", False)
    if with_train_accuracy:
        train_miou_per_epoch = []
    with_16_bit_training = train_d.get("with_16_bit_precision", False)
    starting_test_epoch = train_d.get("starting_test_epoch", num_epochs//2)
    if with_16_bit_training:
        scaler = torch.cuda.amp.GradScaler()
    with_grad_clip = train_d.get("with_grad_clip", False)
    grad_clip_val = train_d.get("grad_clip_val", 1.0)

    if rank == 0 or world_size == 1:
        overall_t1 = time.time()

    if train_d.get("debug_mode", False):
        torch.autograd.set_detect_anomaly(True)

    for epoch_num in range(epoch_num_, num_epochs + 1):
        model.train()
        if world_size > 1:
            train_sampler.set_epoch(epoch_num)
        if rank == 0 or world_size == 1:
            current_lr = round(optims.get_current_lr(optimizer), 7)
            print(f"\nBegin training of epoch {epoch_num} with learning rate {current_lr}.")
            t1 = time.time()
        epoch_loss = 0.0

        # only use tqdm for the process at rank 0
        if rank == 0 or world_size == 1:
            pbar = tqdm.tqdm(total=len(train_dataloader), desc="Training Progress")
        else:
            pbar = None  # Create a dummy pbar for other processes

        for batch_num, data in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)

            inp = data[:-1]

            if preprocess_func is None:
                inp = inp_to_gpu(inp, gpu_num)
            else:
                inp = preprocess_func(inp, gpu_num)
            gt = data[-1].to(gpu_num).reshape(-1)

            # model output and loss computation
            if with_16_bit_training:
                with torch.cuda.amp.autocast():
                    m_output = model(inp)
                    loss = loss_fn(m_output, gt)
            else:
                m_output = model(inp)
                loss = loss_fn(m_output, gt)

            # backprop
            if with_16_bit_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if with_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

            batch_loss = loss.item()
            iteration_losses.append(batch_loss)
            epoch_loss += batch_loss

            if pbar:
                pbar.update(1)
                pbar.set_postfix(loss=batch_loss)

        if pbar:
            pbar.close()

        lrs.step()

        # End of epoch stats
        if rank == 0 or world_size == 1:
            t2 = time.time()
            dif_in_mins = round((t2 - t1) / 60.0, 2)
        ave_epoch_loss = round(epoch_loss / (batch_num + 1), 3)
        if world_size == 1:
            epoch_overall_losses.append(epoch_loss)
            epoch_average_losses.append(ave_epoch_loss)
        else:  # Passing DDP losses to rank 0
            if rank == 0:
                for i in range(1, world_size):
                    epoch_overall_loss_i = torch.zeros(1).to(rank)
                    ave_epoch_loss_i = torch.zeros(1).to(rank)
                    req1 = dist.irecv(tensor=epoch_overall_loss_i, src=i)
                    req1.wait()
                    epoch_loss += epoch_overall_loss_i.cpu().item()
                    req2 = dist.irecv(tensor=ave_epoch_loss_i, src=i)
                    req2.wait()
                    ave_epoch_loss += ave_epoch_loss_i.cpu().item()
                ave_epoch_loss /= world_size # technically an average of averages, but good enough
                epoch_overall_losses.append(epoch_loss)
                epoch_average_losses.append(ave_epoch_loss)
            else:
                epoch_overall_loss_i = torch.tensor(epoch_loss).to(rank)
                ave_epoch_loss_i = torch.tensor(ave_epoch_loss).to(rank)
                req1 = dist.isend(tensor=epoch_overall_loss_i, dst=0)
                req1.wait()
                req2 = dist.isend(tensor=ave_epoch_loss_i, dst=0)
                req2.wait()

        if rank == 0 or world_size == 1:
            print(f"Epoch {epoch_num} is complete after {round(dif_in_mins, 2)} minutes")
            print(f"The average loss for epoch {epoch_num} is {round(ave_epoch_loss, 3)}")
            #print(f"The overall loss for epoch {epoch_num} is {round(epoch_loss, 3)}")

        # logging and model saving
        if rank == 0 or world_size == 1:
            logs_dict = {"epoch_overall_losses": epoch_overall_losses, "epoch_average_losses": epoch_average_losses,
                         "iteration_losses": iteration_losses, "test_miou_vals": test_miou_vals,
                         "test_epochs": test_epochs, "best_test_accuracy": best_test_miou}
            # save the model and optimizer and number of epochs after each epoch
            if world_size == 1:
                model_utils.save_model_during_training(epoch=epoch_num + 1, model=model, optimizer=optimizer,
                                                       lr_scheduler=lrs, logs_dict=logs_dict,
                                                       save_path=model_save_fp)
            else:
                model_utils.save_model_during_training(epoch=epoch_num + 1, model=model.module, optimizer=optimizer,
                                                       lr_scheduler=lrs, logs_dict=logs_dict,
                                                       save_path=model_save_fp)

        if with_train_accuracy:
            train_d["model"] = model
            if rank == 0 or world_size == 1:
                mIoU = part_seg_eval.pc_seg_eval(train_d, dl=train_dataloader, device_id=gpu_num, with_save=False)
                print(f"The mean IoU for epoch {epoch_num} on the training set is {mIoU}.\n")
                train_miou_per_epoch.append(mIoU)
            model.train()

        # validate the model every sampling epoch after the starting epoch
        if test_dataloader:
            if epoch_num >= starting_test_epoch:
                if rank == 0 or world_size == 1:
                    train_d["model"] = model
                    miou = part_seg_eval.pc_seg_eval(train_d, dl=test_dataloader, device_id=gpu_num, with_save=False)
                    print(f"The mean IoU for epoch {epoch_num} on the testing set is {miou} percent.\n")
                    test_epochs.append(epoch_num)
                    test_miou_vals.append(miou)
                    if miou > best_test_miou:
                        best_test_miou = miou
                        if world_size == 1:
                            model_utils.save_model_only_during_training(model, model_save_fp + "_best")
                        else:
                            model_utils.save_model_only_during_training(model.module, model_save_fp + "_best")
                model.train()

    if rank == 0 or world_size == 1:
        overall_t2 = time.time()
        training_time_in_mins = round((overall_t2-overall_t1)/60.0, 2)
        print(f"The overall training time is given by {training_time_in_mins} minutes.")

    # plotting
    if rank == 0 or world_size == 1:
        print("Training is complete.")
        # plot logs fps
        save_fp_epoch_overall_loss = logs_save_fp + "/EpochOverallLosses"
        save_fp_epoch_average_loss = logs_save_fp + "/EpochAverageLosses"
        save_fp_iterations = logs_save_fp + "/Iterations"
        save_fp_test_accuracy = logs_save_fp + "/TestAccuracy"
        if with_train_accuracy:
            save_fp_train_accuracy = logs_save_fp + "/TrainAccuracy"
        # plots
        try:
            plot_utils.plot(x_vals=range(1, num_epochs + 1), y_vals=epoch_overall_losses,
                            title=model_name + "\n Overall Epoch Loss", x_label="Epochs", y_label="Overall Loss",
                            save_fp=save_fp_epoch_overall_loss)
            plot_utils.plot(x_vals=range(1, num_epochs + 1), y_vals=epoch_average_losses,
                            title=model_name + "\n Average Epoch Loss", x_label="Epochs", y_label="Average Loss",
                            save_fp=save_fp_epoch_average_loss)
            plot_utils.plot(x_vals=range(1, len(iteration_losses) + 1), y_vals=iteration_losses,
                            title=model_name + "\n Iterations vs. Loss", x_label="Iterations", y_label="Loss",
                            save_fp=save_fp_iterations)
            plot_utils.plot(x_vals=test_epochs, y_vals=test_miou_vals,
                            title=model_name + "\n Epochs vs. Test MIoU", x_label="Epochs", y_label="Test MIoU",
                            save_fp=save_fp_test_accuracy)
            if with_train_accuracy:
                plot_utils.plot(x_vals=range(1, num_epochs + 1), y_vals=train_miou_per_epoch,
                                title=model_name + "\n Epochs vs. Train MIoU", x_label="Epochs",
                                y_label="Train MIoU",
                                save_fp=save_fp_train_accuracy)
        except Exception as error:
            print(f"Error: {error}")
            print("Error occurred in the plotting, moving on to formal testing.")

    print(f"Deleting dataloaders and shutting down workers...")
    for dl in [train_dataloader, test_dataloader]:
        dist_utils.shutdown_dataloader(dl)

    del train_dataloader
    del test_dataloader
    gc.collect()
    torch.cuda.empty_cache()
