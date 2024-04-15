from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig: # defined for the config components present in artifacts for data ingestion
    # Below are the return types for the components (root_dir is Path format , URL is string etc)
    root_dir : Path 
    source_URL : str
    local_data_file : Path

@dataclass(frozen=True)
class DataValidationConfig: # defined for the config components present in artifacts for data validation
    root_dir : Path
    file_path : Path 
    STATUS_FILE : str
    ALL_REQUIRED_FILES : list