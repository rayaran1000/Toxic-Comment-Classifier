import os
import pickle
import torch
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
        logger.info("Dataset Label Transformation done")

        return dataset_train,dataset_validation

    def tokenize(self,example_batch):

        encodings = self.tokenizer(example_batch['comment_text'].to_list(),max_length=1024,padding=True,truncation=True)

        label_tensors = torch.tensor(example_batch['label'])

        logger.info("Dataset Tokenization done")

        return {'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': label_tensors}

    def convert(self):
        train_dataset , validation_dataset = self.transform()

        #Tokenizing the datasets
        train_dataset_tokenized = self.tokenize(train_dataset)
        validation_dataset_tokenized = self.tokenize(validation_dataset)

        #Saving the tokenized datasets
        train_tokenized_path = os.path.join(self.config.root_dir , 'train_tokenized.pt')
        validation_tokenized_path = os.path.join(self.config.root_dir , 'validation_tokenized.pt')

        torch.save(train_dataset_tokenized, train_tokenized_path)
        torch.save(validation_dataset_tokenized, validation_tokenized_path)
        logger.info("Dataset Transformation done")