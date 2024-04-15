from ToxicCommentClassifier.logging import logger
from ToxicCommentClassifier.utils.common import get_size
from datasets import load_dataset
import os

from pathlib import Path
from ToxicCommentClassifier.entity import DataIngestionConfig
import pickle

class DataIngestion:
    def __init__(self,config:DataIngestionConfig): # It will take the configuration from DataIngestionConfig defined earlier , which will in turn use Configuration Manager to take data from config.yaml
        self.config = config

    def load_save_file(self):
        if not os.path.exists(self.config.local_data_file): # If file does not exist
            dataset = load_dataset(self.config.source_URL)
            logger.info(f"{self.config.source_URL} : loaded from hugging face")
            with open(self.config.local_data_file, "wb") as f:
                pickle.dump(dataset, f)

        
        else:
            logger.info(f"File already exists of size : {get_size(Path(self.config.local_data_file))}") # Checking file size present already in the path
        