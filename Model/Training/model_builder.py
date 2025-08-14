import point_transformer.model.pointtransformer.pointtransformer_seg as pointtransformer_seg

def get_model(model_cfg: dict):
    model_name = model_cfg["model_name"]
    if model_name == "PointTransformer":
        model = pointtransformer_seg.from_cfg(model_cfg)
    else:
        raise ValueError(f"Model type {model_name} not Supported")
    return model


def get_model_from_config(model_cfg: dict, train_d: dict) -> None:
    train_d["model"] = get_model(model_cfg)
    train_d["model_name"] = model_cfg["model_name"]
    train_d["task"] = model_cfg["task"]
