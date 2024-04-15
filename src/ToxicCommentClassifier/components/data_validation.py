import pickle
from pathlib import Path
from ToxicCommentClassifier.logging import logger
from ToxicCommentClassifier.utils.common import get_size

from ToxicCommentClassifier.entity import DataValidationConfig

class DataValidation:
    def __init__(self,config:DataValidationConfig): # It will take the configuration from DataIngestionConfig defined earlier , which will in turn use Configuration Manager to take data from config.yaml
        self.config = config

    def validation_all_files_exist(self) -> bool:

        try:
            validation_status = None # Validation status counter for checking all files exists or not

            if get_size(Path(self.config.file_path)) != '0 KB' : # If File is not empty, we will load the dataset
                with open(self.config.file_path, "rb") as f:
                    dataset = pickle.load(f)
                    logger.info("Dataset loaded successfully for validation")
                

            for file in self.config.ALL_REQUIRED_FILES:
                if file not in dataset.keys(): # checking whether all the files mentioned in ALL_REQUIRED_FILES is present or not in the dataset file
                    validation_status = False
                    with open(self.config.STATUS_FILE,'a') as f:
                        f.write(f"Validation Status : {validation_status} - file : {file}\n")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE,'a') as f:
                        f.write(f"Validation Status : {validation_status} - file : {file}\n")
                
            return validation_status
        
        except Exception as e:
            raise e