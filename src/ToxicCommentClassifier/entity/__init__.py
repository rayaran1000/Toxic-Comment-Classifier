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
    tokenizer_name : Path