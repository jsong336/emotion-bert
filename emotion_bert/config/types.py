import pathlib
from typing import Union, Literal, Any
from dataclasses import dataclass

@dataclass
class ResourceDatasetConfig:
    dir: pathlib.Path

@dataclass
class ResourceConfig:
    device: Literal["cuda", "cpu"]
    dataset: ResourceDatasetConfig

@dataclass
class DataDatasetTypeConfig:
    file: str 
    columns: list[str]

@dataclass
class DataDatasetConfig:
    train: DataDatasetTypeConfig
    test: DataDatasetTypeConfig

@dataclass
class BertFlatConfig:
    sentence_max_len: int
    pretrained_encoder: str
    fc_hiddens: list[int]
    dropout_p: float 
    threshold: float 

ModelConfig = Union[dict[str, Any], BertFlatConfig]

@dataclass
class TrainValidationConfig:
    split: float 
    step: int 
    batch: int

@dataclass 
class TrainTrainConfig:
    batch: int 
    epoch: int 
    learning_rate: float
    grad_clip: float
    weight_decay: float
    warmup_ratio: float 

@dataclass
class TrainCheckpointConfig:
    dir: str
    step: int
    
@dataclass
class TrainConfig:
    seed: float 
    validation: TrainValidationConfig
    train: TrainTrainConfig
    checkpoint: TrainCheckpointConfig

    
@dataclass
class DataConfig:
    dir: pathlib.Path
    extra: dict[str, Any]
    dataset: DataDatasetConfig
    
@dataclass
class Config:
    model_name: str # model name
    config_path: pathlib.Path # model config path
    resource: ResourceConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
