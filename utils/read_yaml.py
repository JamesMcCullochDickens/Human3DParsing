import yaml


def yml_to_dict(yml_fp: str) -> dict:
    with open(yml_fp, "r") as stream:
        try:
            y = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return y