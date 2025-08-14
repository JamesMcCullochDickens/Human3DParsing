import warnings
warnings.filterwarnings('ignore')
import os
os.environ["cwd"] = os.getcwd()
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging
import numpy as np
np.set_printoptions(suppress=True)
import torch
torch.set_printoptions(sci_mode=False)
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import ast

import utils.path_utils as path_utils
import utils.input_utils as input_utils
import utils.cache_dict_utils as cache_dict_utils
import Training.model_builder as model_builder
import utils.read_yaml as read_yaml
import Training.train_builder as train_builder
import Training.pc_seg_train as pc_seg_train
import Training.model_utils as model_utils
import Evaluation.pc_seg_eval as pc_seg_eval


model_configs_outer_path = os.path.join(os.getcwd(), "Configs/Model_Configs")
os.makedirs(model_configs_outer_path, exist_ok=True)
train_configs_outer_path = os.path.join(os.getcwd(), "Configs/Train_Configs")
os.makedirs(train_configs_outer_path, exist_ok=True)


def create_artifact_paths(model_dir: str, with_del: bool,
                          outer_path: str = os.getcwd()) -> dict:
    # Trained models
    trained_models_outer_fp = os.path.join(outer_path, "Trained_Models")
    trained_models_fp = os.path.join(outer_path, "Trained_Models", model_dir)
    path_utils.create_if_not_exists(trained_models_outer_fp, with_delete=False)
    path_utils.create_if_not_exists(trained_models_fp, with_delete=with_del)

    # Training logs
    training_logs_outer_fp = os.path.join(outer_path, "Training_Logs")
    training_logs_fp = os.path.join(outer_path, "Training_Logs", model_dir)
    path_utils.create_if_not_exists(training_logs_outer_fp, with_delete=False)
    path_utils.create_if_not_exists(training_logs_fp, with_delete=with_del)

    # Eval results
    outer_eval_results_fp = os.path.join(outer_path, "Eval_Results")
    eval_results_fp = os.path.join(outer_path, "Eval_Results", model_dir)
    path_utils.create_if_not_exists(outer_eval_results_fp, with_delete=False)
    path_utils.create_if_not_exists(eval_results_fp, with_delete=with_del)

    # Saved Predictions
    outer_pred_fp = os.path.join(outer_path, "Predictions")
    pred_fp = os.path.join(outer_pred_fp,  model_dir)
    path_utils.create_if_not_exists(outer_pred_fp, with_delete=False)
    path_utils.create_if_not_exists(pred_fp, with_delete=with_del)


    return {"trained_models_fp": trained_models_fp,
            "training_logs_fp": training_logs_fp,
            "eval_results_fp": eval_results_fp,
            "pred_fp": pred_fp}


def train_and_test(d: dict) -> None:
    task = d["task"]
    if task == "human_3d_pc_semantic_segmentation":
        if not d["only_test"]:
            pc_seg_train.train_launch(d)
        if d["with_test"]:
            best_model_save_fp = os.path.join(d["save_fps"]["trained_models_fp"], d["model_name"] + "_best")
            if os.path.exists(best_model_save_fp):
                model = model_utils.load_model_from_save(best_model_save_fp, d["model"])
            else:
                print(f"No best model, loading last checkpoint.")
                best_model_save_fp = os.path.join(d["save_fps"]["trained_models_fp"], d["model_name"])
                model = model_utils.load_model_from_save(best_model_save_fp, d["model"])
            d["model"] = model
            if "gpu_override" in d:
                rank = d["gpu_override"][0]
            else:
                rank = 0
            pc_seg_eval.pc_seg_eval_formal(d, d["test_dl"], device_id=rank, save_pred=True)


def parse_indices(indices_str: str) -> List[int]:
    try:
        indices = ast.literal_eval(indices_str)
        if not isinstance(indices, list):
            raise argparse.ArgumentTypeError('Indices must be in list format, e.g., [1,2,3]')
        if not all(isinstance(idx, int) for idx in indices):
            raise argparse.ArgumentTypeError('Indices must be integers')
        return indices
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError('Invalid format for indices. Please use list format, e.g., [1,2,3]')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parsing arguments for the run.py file.")
    parser.add_argument("model_config_name", type=str, help="Name of the model configuration")
    parser.add_argument("train_config_name", type=str, help="Name of the train configuration")
    parser.add_argument("--with_load", action="store_true", help="Include if you want to load a model")
    parser.add_argument("--only_test", action="store_true", help="Include if you only want to perform testing")
    parser.add_argument("--gpu_override", nargs='?', type=parse_indices, help='List of gpus in list format, '
                                                                              'e.g., [0,1,2], without whitespace, or surrounded'
                                                                              'by quotes.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    model_config_name = args.model_config_name
    train_config_name = args.train_config_name
    only_test = args.only_test
    with_load = args.with_load or only_test
    gpu_override = args.gpu_override

    # Build model from config
    print("Building model from config.")
    model_cfg_path = os.path.join(model_configs_outer_path, model_config_name)
    assert os.path.exists(model_cfg_path), f"No such model config path {model_cfg_path} exists"
    model_cfg = read_yaml.yml_to_dict(model_cfg_path)

    # Create train_builder from config
    print("Loading the training config and dataloader...")
    train_d = {}
    train_config_path = os.path.join(train_configs_outer_path, train_config_name)
    assert os.path.exists(train_config_path), f"No such train config path {train_config_path} exists"
    train_config = read_yaml.yml_to_dict(train_config_path)
    model_builder.get_model_from_config(model_cfg, train_d)
    print("Model built from config.")

    overall_cfg = {**model_cfg, **train_config}


    # Create artefact paths
    model_save_dir = model_config_name[:-4] + "_" + train_config_name[:-4]
    with_delete = not with_load
    trained_models_dir = os.path.join(os.getcwd(), "data", "models", model_save_dir)
    if os.path.exists(trained_models_dir) and with_delete:
        prompt = f"Model {model_save_dir} has a directory, are you sure you would like to delete it? If no, " \
                 f"with_load will automatically be set to true.\nEnter yes or no: "
        inp = input_utils.get_input_with_timeout(prompt=prompt, timeout=10, default="yes")
        if inp in ["No", "n", "N", "no"]:
            with_delete = False
            with_load = True

    # Loading and testing info
    train_builder.train_build_from_cfg(train_config, train_d)
    train_d["with_test"] = train_config.get("with_test", True)
    train_d["model_name"] = model_config_name[:-4]
    train_d["with_load"] = with_load
    train_d["only_test"] = only_test
    train_d["save_fps"] = create_artifact_paths(
                                    model_dir=model_save_dir,
                                    with_del=not with_load,
                                    outer_path=os.path.join(os.getcwd(), "data", "models")
                                                )

    # Save configs as an artefact in the training logs
    configs_save_fp = os.path.join(train_d["save_fps"]["training_logs_fp"], "model_train_cfg.json")
    cache_dict_utils.write_readable_cached_dict(configs_save_fp, overall_cfg)

    print("Finished loading the training config and dataloader.")
    print(f"Proceeding with model {model_config_name} and train config {train_config_name}")
    if gpu_override:
        train_d["gpu_override"] = gpu_override
    train_and_test(train_d)
