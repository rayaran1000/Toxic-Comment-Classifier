from ToxicCommentClassifier.config.configuration import ConfigurationManager
from ToxicCommentClassifier.components.data_validation import DataValidation


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config() # Storing the configuration
            data_validation = DataValidation(config=data_validation_config) # Using the configuration saved earlier to call data_ingestion
            data_validation.validation_all_files_exist()
            
        except Exception as e:
            raise e