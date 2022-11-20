import pathlib
import yaml
from argparse import ArgumentParser, Namespace
from .types import *


def resolve_config_path(args: Namespace) -> pathlib.Path:
    config_dir: pathlib.Path = (
        pathlib.Path(args.config_dir)
        if args.config_dir
        else pathlib.Path(__package__).resolve().parent / "config"
    )

    config_path: str = args.config_path
    if not args.config_path:
        config_path = config_dir / (args.model_name + ".yaml")

    if not config_path.exists():
        # if path doesn't exists, check again with yml extensions
        config_path = pathlib.Path(str(config_path).replace(".yaml", ".yml"))

    if not config_path.exists():
        raise ValueError(f"{config_path} does not exists.")

    return config_path


def load_config_arg_dict() -> dict[str, Any]:
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("--config_dir", type=str, required=False)
    parser.add_argument("--config_path", type=str, required=False)

    args = parser.parse_args()

    config_path = resolve_config_path(args)

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dict["model_name"] = args.model_name
    config_dict["config_path"] = config_path

    return config_dict


def load_config() -> Config:
    c = load_config_arg_dict()

    return Config(
        model_name=c.get("model_name"),
        config_path=c.get("config_path"),
        resource=ResourceConfig(
            device=c.get("resource", {}).get("device", "gpu"),
            dataset=ResourceDatasetConfig(**c.get("resource", {}).get("dataset", {})),
        ),
        data=DataConfig(
            dir=c.get("data", {}).get("dir"),
            extra=c.get("data", {}).get("extra"),
            dataset=DataDatasetConfig(
                train=DataDatasetTypeConfig(
                    **(c.get("data", {}).get("dataset", {}).get("train", {}))
                ),
                test=DataDatasetTypeConfig(
                    **(c.get("data", {}).get("dataset", {}).get("test", {}))
                ),
            ),
        ),
        model=c.get("model", {}),
        train=TrainConfig(
            seed=c.get("train", {}).get("seed", 0),
            validation=TrainValidationConfig(
                **c.get("train", {}).get("validation", {})
            ),
            train=TrainTrainConfig(**c.get("train", {}).get("train", {})),
            checkpoint=TrainCheckpointConfig(
                **c.get("train", {}).get("checkpoint", {})
            ),
        ),
    )
