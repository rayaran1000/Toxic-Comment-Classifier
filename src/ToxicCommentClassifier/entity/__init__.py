from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig: # defined for the config components present in artifacts for data ingestion
    root_dir : Path 
    source_URL : str
    local_data_file : Path

@dataclass(frozen=True)
class DataValidationConfig: # defined for the config components present in artifacts for data validation
    root_dir : Path
    file_path : Path 
    STATUS_FILE : str
    ALL_REQUIRED_FILES : list

@dataclass(frozen=True)
class DataTransformationConfig: # defined for the config components present in artifacts for data transformation
    root_dir : Path 
    data_path : Path

@dataclass(frozen=True)
class ModelTrainerConfig: # defined for the config components present in artifacts for model training
    root_dir : Path 
    data_path_train : Path
    data_path_validation : Path
    tokenizer_ckpt : Path
    model_ckpt : Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    eval_steps: int
    save_steps: int
    save_total_limit: int

@dataclass(frozen=True)
class PredictionConfig: # defined for the config components present in artifacts for prediction pipeline
    finetuned_model_path: Path
    tokenizer_ckpt: Path