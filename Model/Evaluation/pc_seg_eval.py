import os
from typing import *
import shutil

import glfw
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode
import tqdm
import pprint
import numpy as np


import utils.cache_dict_utils as cached_dict_utils
import Training.pc_seg_train as pc_seg_train
import utils.pc_vis as pc_vis
import utils.path_utils as path_utils
import utils.frame_utils as frame_utils
import Dataloaders.human_seg_dl as human_seg_dl
import Data_Processing.src.PyOpenGL.render_part_pc_4view as render_4view
import data_fps
import utils.hdf5_utils as hdf5_utils
import knn_cuda

outer_pc_seg_gt_data_fp = data_fps.data_fps["Human_Seg_PC_Data"]


def to_pct(f: float) -> float:
    return round(100 * f, 1)


@torch.no_grad()
def pc_seg_eval(d: dict, dl: torch.utils.data.DataLoader,
             device_id: int = 0, with_save: bool = False) -> Union[None, float]:
    model = d["model"]
    cm_nts = d["cm_num_to_str"]
    preprocess_func = d["preprocess_func"]

    model.eval()
    model.to(device_id)

    eval_results = {}
    for key in cm_nts.keys():
        eval_results[cm_nts[key]] = {"tp": 0, "fp": 0, "fn": 0}

    num_ims = 0
    cat_keys = list(cm_nts.keys())

    for data in tqdm.tqdm(dl):
        inp = data[:-1]

        if preprocess_func is None:
            inp = pc_seg_train.inp_to_gpu(inp, device_id)
        else:
            inp = preprocess_func(inp, device_id)
        gt = data[-1].to(device_id)
        batch_size, num_points = gt.shape

        # model prediction
        output = model(inp)
        pred = torch.argmax(output, dim=-1)
        pred = pred.reshape(batch_size, num_points)

        # class-wise eval
        unique_pred = torch.unique(pred).cpu().numpy().tolist()
        unique_gt = torch.unique(gt).cpu().numpy().tolist()
        unique_pred_s = set(unique_pred)
        unique_gt_s = set(unique_gt)
        all_vals = unique_pred_s.union(unique_gt_s)

        for val in all_vals:
            gt_val = int(val)

            gt = gt.reshape(-1)
            pred = pred.reshape(-1)

            pred_eq_mask = (pred == val)
            pred_neq_mask = ~ pred_eq_mask
            gt_eq_mask = (gt == val)
            gt_neq_mask = ~ gt_eq_mask

            tp = torch.sum(torch.logical_and(gt_eq_mask, pred_eq_mask)).item()
            fn = torch.sum(torch.logical_and(gt_eq_mask, pred_neq_mask)).item()
            fp = torch.sum(torch.logical_and(gt_neq_mask, pred_eq_mask)).item()
            eval_results[cm_nts[gt_val]]["tp"] += tp
            eval_results[cm_nts[gt_val]]["fn"] += fn
            eval_results[cm_nts[gt_val]]["fp"] += fp

    # Prediction results
    eval_results["num_images"] = num_ims
    valid_cats = 0  # categories with at least 1 ground truth pixel
    all_pixels = 0
    correct = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        union = fp + tp + fn
        num_pix = tp + fn
        if num_pix == 0:
            eval_results[cm_nts[key]]["IoU"] = "None"
        else:
            valid_cats += 1
            eval_results[cm_nts[key]]["IoU"] = to_pct(tp / union)
            eval_results[cm_nts[key]]["num_points"] = num_pix
            all_pixels += num_pix
        correct += tp

    eval_results["acc"] = to_pct(correct / all_pixels)
    iou_sum = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        if eval_results[cm_nts[key]]["IoU"] != "None":
            union = fp + tp + fn
            iou_sum += tp / union

    mIoU = iou_sum / valid_cats
    eval_results["mIoU"] = to_pct(mIoU)

    if not with_save:
        return to_pct(mIoU)
    else:
        eval_save_fp = d["save_fps"]["eval_results_fp"]
        print(f"Evaluation complete...\n")
        pprint.pprint(eval_results)
        results_fp = os.path.join(eval_save_fp, "class_results.txt")
        cached_dict_utils.write_readable_cached_dict(results_fp, eval_results)


@torch.no_grad()
def pc_seg_eval_formal(d: dict, dl: torch.utils.data.DataLoader,
             device_id: int = 0, save_pred: bool = True, with_render: bool = True,
                       with_orig_res_eval: bool = True) -> Union[None, float]:
    model = d["model"]
    cm_nts = d["cm_num_to_str"]
    preprocess_func = d["preprocess_func"]

    model.eval()
    model.to(device_id)

    eval_results = {}
    for key in cm_nts.keys():
        eval_results[cm_nts[key]] = {"tp": 0, "fp": 0, "fn": 0}

    cat_keys = list(cm_nts.keys())

    if save_pred:
        try:
            file_names = dl.dataset.data_keys
        except:
            file_names = list(range(len(dl.dataset)))
        pred_d = {}
        counter = 0

    for data in tqdm.tqdm(dl):
        inp = data[:-1]

        if preprocess_func is None:
            inp = pc_seg_train.inp_to_gpu(inp, device_id)
        else:
            inp = preprocess_func(inp, device_id)
        gt = data[-1].to(device_id)
        batch_size, num_points = gt.shape

        # model prediction
        output = model(inp)
        pred = torch.argmax(output, dim=-1)
        pred = pred.reshape(batch_size, num_points)

        if save_pred:
            for p in pred:
                file_name = file_names[counter]
                pred_d[file_name] = p.cpu()
                counter += 1

        # class-wise eval
        unique_pred = torch.unique(pred).cpu().numpy().tolist()
        unique_gt = torch.unique(gt).cpu().numpy().tolist()
        unique_pred_s = set(unique_pred)
        unique_gt_s = set(unique_gt)
        all_vals = unique_pred_s.union(unique_gt_s)

        for val in all_vals:
            gt_val = int(val)

            gt = gt.reshape(-1)
            pred = pred.reshape(-1)

            pred_eq_mask = (pred == val)
            pred_neq_mask = ~ pred_eq_mask
            gt_eq_mask = (gt == val)
            gt_neq_mask = ~ gt_eq_mask

            tp = torch.sum(torch.logical_and(gt_eq_mask, pred_eq_mask)).item()
            fn = torch.sum(torch.logical_and(gt_eq_mask, pred_neq_mask)).item()
            fp = torch.sum(torch.logical_and(gt_neq_mask, pred_eq_mask)).item()
            eval_results[cm_nts[gt_val]]["tp"] += tp
            eval_results[cm_nts[gt_val]]["fn"] += fn
            eval_results[cm_nts[gt_val]]["fp"] += fp

    # Prediction results
    valid_cats = 0  # categories with at least 100 ground truth pixels
    all_pixels = 0
    correct = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        union = fp + tp + fn
        num_pix = tp + fn
        if num_pix < 100:
            eval_results[cm_nts[key]]["IoU"] = "None"
        else:
            valid_cats += 1
            eval_results[cm_nts[key]]["IoU"] = to_pct(tp / union)
            eval_results[cm_nts[key]]["num_points"] = num_pix
            all_pixels += num_pix
        correct += tp

    # accuracy
    eval_results["acc"] = to_pct(correct / all_pixels)

    # frequency weighted accuracy
    fw_acc = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        num_pix = tp + fn
        if num_pix > 100:
            acc_c = tp / num_pix
            weight_c = num_pix / all_pixels
            fw_acc += weight_c * acc_c
    eval_results["fwAcc"] = to_pct(fw_acc)

    # mean IoU
    iou_sum = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        if eval_results[cm_nts[key]]["IoU"] != "None":
            union = fp + tp + fn
            iou_sum += tp / union
    mIoU = iou_sum / valid_cats
    eval_results["mIoU"] = to_pct(mIoU)

    # frequency weighted mean iou
    wIoU = 0
    for key in cat_keys:
        fp, tp, fn = (eval_results[cm_nts[key]]["fp"], eval_results[cm_nts[key]]["tp"],
                      eval_results[cm_nts[key]]["fn"])
        if eval_results[cm_nts[key]]["IoU"] != "None":
            union = fp + tp + fn
            iou_per_class = tp / union
            class_count = eval_results[cm_nts[key]]["num_points"]
            weight_iou_class = (class_count/all_pixels) * iou_per_class
            wIoU += weight_iou_class
    eval_results["wIoU"] = to_pct(wIoU)

    # results dict
    eval_save_fp = d["save_fps"]["eval_results_fp"]
    print(f"Evaluation complete...\n")
    pprint.pprint(eval_results)
    results_fp = os.path.join(eval_save_fp, "class_results.txt")
    cached_dict_utils.write_readable_cached_dict(results_fp, eval_results)

    # Simple summary
    summary_fp = os.path.join(eval_save_fp, "summary.txt")
    with open(summary_fp, "w") as f:
        f.write("Overall Accuracy " + str(eval_results["acc"])+"\n")
        f.write("Frequency Weighed Accuracy " + str(eval_results["fwAcc"]) + "\n")
        f.write("Mean IoU " + str(eval_results["mIoU"]) + "\n")
        f.write("Frequency Weighted Mean IoU " + str(eval_results["wIoU"]))


    if save_pred:
        pred_save_fp = os.path.join(eval_save_fp, "pred.pt")
        torch.save(pred_d, pred_save_fp)

    with_sapiens = "cihp" not in dl.dataset.dataset_name
    if with_sapiens:
        is_v2 = d["is_sapiens_v2"]
        if is_v2:
            num_parts = 18
            label_names = human_seg_dl.sapiens_v2_cm_str_to_num
        else:
            label_names = human_seg_dl.sapiens_v1_cm_str_to_num
            num_parts = 15
    else:
        label_names =  human_seg_dl.cihp_cm_num_to_str
        is_v2 = False
        num_parts = 20

    num_colors = num_parts

    window=None
    # rendering with ground truth labels versus predicted labels
    if with_render and save_pred:
        render_outer_fp = os.path.join(eval_save_fp, "4view_renders")
        os.makedirs(render_outer_fp, exist_ok=True)
        counter = 0
        for data in tqdm.tqdm(dl, total=len(dl)):
            pcd = data[0].numpy()
            gt_labels = data[1].numpy()
            batch_size, num_points = gt_labels.shape
            num_channels = pcd.shape[-1]
            pcd = pcd.reshape(batch_size, num_points, num_channels)[..., :3]
            for batch_num in range(batch_size):
                vertices = pcd[batch_num]
                file_name = file_names[counter]
                pred_labels = pred_d[file_name].numpy()
                gt_labels_b = gt_labels[batch_num]
                pred_ims = []
                for i, labels in enumerate([gt_labels_b, pred_labels]):
                    vertex_label_colors = frame_utils.get_color_map(num_vals=num_colors)
                    part_colors = pc_vis.parts_to_colors(labels, vertex_label_colors)
                    part_colors_norm = part_colors.astype(float) / 255.0
                    label_d = {}
                    for i, label_name in enumerate(label_names.items()):
                        label_d[label_name] = vertex_label_colors[i].astype(float) / 255.0
                    vis_d = {"points": vertices, "colors": part_colors_norm}
                    view, window = render_4view.render_pcd_4view(vis_d, width=2000, height=2000,
                                                         show_window=False, window=window,
                                                         destroy_window=False)
                    view4_im = frame_utils.concatenate_images_horizontally([v for v in view])
                    pred_ims.append(view4_im)

                overall_pred_comparison_im = frame_utils.concatenate_images_vertically([*pred_ims])
                views_save_fp = os.path.join(render_outer_fp, str(file_name)+".jpg")
                frame_utils.get_legend_with_im(overall_pred_comparison_im, color_labels=label_d,
                                                                save_fp=views_save_fp,
                                                                legend_scale=3.0)
                counter += 1

        glfw.destroy_window(window)
        glfw.terminate()

        print(f"Zipping 4view gt/pred renders to {render_outer_fp}.zip")
        shutil.make_archive(render_outer_fp, 'zip', render_outer_fp)

    # Nearest neighbor upsampling
    if with_orig_res_eval:
        d["pred_save_fp"] = pred_save_fp
        num_neighbors = d.get("num_neighbors", 3)
        nn_upsample_save_fp = os.path.join(eval_save_fp, "nn_upsample_eval")
        os.makedirs(nn_upsample_save_fp, exist_ok=True)
        d["nn_outer_save_fp"] = nn_upsample_save_fp
        nn_upsample_eval(d, num_neighbors)


def fast_majority_vote(labels: torch.Tensor, num_classes: int):
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)  # (N, k, num_classes)
    votes = one_hot.sum(dim=1)  # (N, num_classes)
    return votes.argmax(dim=1)


@torch.no_grad()
def nn_upsample_eval(d: dict, num_neighbors: int = 3):
    pred_save_fp = d["pred_save_fp"]
    sampled_dataset_vertices_fp = d["sampled_dataset_vertices_fp"]
    original_dataset_vertices_fp = d["original_dataset_vertices_fp"]
    original_dataset_labels_fp = d["original_dataset_labels_fp"]
    outer_save_fp = d["nn_outer_save_fp"]
    cm_nts = d["cm_num_to_str"]
    cm_str_to_num = d["cm_str_to_num"]
    renders_outer_fp = os.path.join(outer_save_fp, "renders")
    os.makedirs(renders_outer_fp, exist_ok=True)
    num_parts = len(cm_nts.keys())

    label_d = {}
    vertex_label_colors = frame_utils.get_color_map(num_vals=num_parts)
    for i, label_name in enumerate(cm_str_to_num.items()):
        label_d[label_name] = vertex_label_colors[i].astype(float) / 255.0

    eval_results = {}
    for key in cm_nts.keys():
        eval_results[cm_nts[key]] = {"tp": 0, "fp": 0, "fn": 0}

    cat_keys = list(cm_nts.keys())

    d_pred = torch.load(pred_save_fp)
    all_pixels = 0
    correct = 0
    valid_cats = 0

    for pc_num, key in tqdm.tqdm(enumerate(d_pred.keys()), total=len(d_pred.keys())):
        pred = d_pred[key].numpy()
        original_pcd = hdf5_utils.load_entry_by_key(original_dataset_vertices_fp, key)
        original_labels = hdf5_utils.load_entry_by_key(original_dataset_labels_fp, key)
        sampled_pcd = hdf5_utils.load_entry_by_key_nested(sampled_dataset_vertices_fp, key)["sampled_vertices"]

        """
        # Scikit learn version, slow, but no requirement for cuda
        nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(sampled_pcd)
        distances, indices = nn.kneighbors(original_pcd)
        neighbor_labels = pred[indices]
        extrapolated_preds = fast_majority_vote(torch.tensor(neighbor_labels), num_classes=num_parts).numpy()
        """

        # Cuda version
        sampled_pcd_t = torch.unsqueeze(torch.tensor(sampled_pcd), 0).to(0)
        original_pcd_t = torch.unsqueeze(torch.tensor(original_pcd), 0).to(0)
        indices = knn_cuda.knn_query(original_pcd_t, sampled_pcd_t, num_neighbors)[0].cpu().numpy()
        neighbor_labels = pred[indices]
        extrapolated_preds = fast_majority_vote(torch.tensor(neighbor_labels), num_classes=num_parts).cpu().numpy()

        # Evaluation
        gt = torch.tensor(original_labels).reshape(-1)
        pred_labels = torch.tensor(extrapolated_preds).reshape(-1)

        for val in set(gt.tolist()).union(set(pred_labels.tolist())):
            val = int(val)
            pred_eq_mask = (pred_labels == val)
            pred_neq_mask = ~pred_eq_mask
            gt_eq_mask = (gt == val)
            gt_neq_mask = ~gt_eq_mask

            tp = torch.sum(gt_eq_mask & pred_eq_mask).item()
            fn = torch.sum(gt_eq_mask & pred_neq_mask).item()
            fp = torch.sum(gt_neq_mask & pred_eq_mask).item()

            eval_results[cm_nts[val]]["tp"] += tp
            eval_results[cm_nts[val]]["fn"] += fn
            eval_results[cm_nts[val]]["fp"] += fp

        correct += torch.sum(gt == pred_labels).item()
        all_pixels += gt.shape[0]

        # Render

        original_labels_colors = pc_vis.parts_to_colors(original_labels, vertex_label_colors)
        original_labels_colors_norm = original_labels_colors.astype(float) / 255.0
        upsampled_part_colors = pc_vis.parts_to_colors(extrapolated_preds, vertex_label_colors)
        upsampled_part_colors_norm = upsampled_part_colors.astype(float) / 255.0

        vis_pred_d = {"points": original_pcd, "colors": upsampled_part_colors_norm}
        vis_gt_d = {"points": original_pcd, "colors": original_labels_colors_norm}

        view4_im_pred = render_4view.render_pcd_4view(vis_pred_d, width=2000, height=2000, show_window=False, destroy_window=True)
        view4_im_pred = frame_utils.concatenate_images_horizontally(view4_im_pred)
        view4_im_gt = render_4view.render_pcd_4view(vis_gt_d, width=2000, height=2000, show_window=False, destroy_window=True)
        view4_im_gt = frame_utils.concatenate_images_horizontally(view4_im_gt)

        all_ims = [view4_im_pred, view4_im_gt]
        overall_pred_comparison_im = frame_utils.concatenate_images_vertically(all_ims)
        views_save_fp = os.path.join(renders_outer_fp, key + ".jpg")
        frame_utils.get_legend_with_im(overall_pred_comparison_im, color_labels=label_d,
                                       save_fp=views_save_fp, legend_scale=3.0)


    # Metric summaries
    iou_sum = 0
    fw_acc = 0
    fw_iou = 0
    for key in cat_keys:
        stats = eval_results[cm_nts[key]]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        num_pix = tp + fn
        union = tp + fp + fn
        if num_pix > 100:
            iou = tp / union if union > 0 else 0.0
            acc = tp / num_pix
            weight = num_pix / all_pixels

            stats["IoU"] = to_pct(iou)
            stats["num_points"] = num_pix
            iou_sum += iou
            fw_acc += acc * weight
            fw_iou += iou * weight
            valid_cats += 1
        else:
            stats["IoU"] = "None"

    eval_results["acc"] = to_pct(correct / all_pixels)
    eval_results["fwAcc"] = to_pct(fw_acc)
    eval_results["mIoU"] = to_pct(iou_sum / valid_cats if valid_cats > 0 else 0)
    eval_results["wIoU"] = to_pct(fw_iou)

    print(f"Upsampled Evaluation complete...\n")
    pprint.pprint(eval_results)

    results_fp = os.path.join(outer_save_fp, "class_results.txt")
    cached_dict_utils.write_readable_cached_dict(results_fp, eval_results)

    summary_fp = os.path.join(outer_save_fp, "summary.txt")
    with open(summary_fp, "w") as f:
        f.write("Overall Accuracy: " + str(eval_results["acc"]) + "\n")
        f.write("Frequency Weighted Accuracy: " + str(eval_results["fwAcc"]) + "\n")
        f.write("Mean IoU: " + str(eval_results["mIoU"]) + "\n")
        f.write("Frequency Weighted Mean IoU: " + str(eval_results["wIoU"]) + "\n")


if __name__ == "__main__":
    ...