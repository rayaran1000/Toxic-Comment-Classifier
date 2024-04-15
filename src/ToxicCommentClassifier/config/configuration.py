from ToxicCommentClassifier.constants import *
from ToxicCommentClassifier.utils.common import read_yaml,create_directories
from ToxicCommentClassifier.entity import DataIngestionConfig
from ToxicCommentClassifier.entity import DataValidationConfig

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
    # Here we are reading the yaml file and we can now use the file paths present inside pararms and config.yaml        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root]) # Here we are calling the artifacts_root key values using '.' , which was the purpose of @ensure_annotations

    def get_data_ingestion_config(self) -> DataIngestionConfig: # Here we are using the entity to specify the return type classes to make sure proper output is returned
        config= self.config.data_ingestion # Calling the data_ingestion dictionary created in config.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        data_ingestion_config = DataIngestionConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig: # Here we are using the entity to specify the return type classes to make sure proper output is returned
        config= self.config.data_validation # Calling the data_validation dictionary created in config.yaml file

        create_directories([config.root_dir]) # Creating a directory using the root directory

        data_validation_config = DataValidationConfig( # Extracting the values from the config.yaml to here inside data_ingestion_config
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            file_path=config.file_path,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )

        return data_validation_config