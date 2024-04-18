import os
import pickle
import pandas as pd
from ToxicCommentClassifier.logging import logger
from transformers import AutoTokenizer
from ToxicCommentClassifier.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self,config:DataTransformationConfig): # It will take the configuration from DataIngestionConfig defined earlier , which will in turn use Configuration Manager to take data from config.yaml
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def load_dataset(self): # Used for tokenization of input text field

        # Loading the dataset
        with open(self.config.data_path, "rb") as f:
            dataset = pickle.load(f)
            return dataset
        
    def transform(self):

        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        dataset = self.load_dataset()

        # Creating seperate train and validation datasets and concatenating the label columns into a list
        dataset_train = pd.DataFrame(dataset['train'])
        dataset_validation = pd.DataFrame(dataset['validation'])
        
        dataset_train['label'] = dataset_train[label_columns].astype('float32').apply(list, axis=1)
        dataset_train = dataset_train[['comment_text','label']]
        
        dataset_validation['label'] = dataset_validation[label_columns].astype('float32').apply(list, axis=1)
        dataset_validation = dataset_validation[['comment_text','label']]
        logger.info("Data Transformation done")

        #Saving the transformed datasets
        train_transformed_path = os.path.join(self.config.root_dir , 'train_transformed')
        validation_transformed_path = os.path.join(self.config.root_dir , 'validation_transformed')

        dataset_train.to_csv(train_transformed_path,index=False)
        dataset_validation.to_csv(validation_transformed_path,index=False)
         