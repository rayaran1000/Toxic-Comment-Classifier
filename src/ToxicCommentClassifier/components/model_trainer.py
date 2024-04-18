import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
from transformers import EvalPrediction

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments


from ToxicCommentClassifier.entity import ModelTrainerConfig

# Multi-Label Classification Evaluation Metrics
class MultiLabelMetric:
  def multi_labels_metrics(self,predictions, labels, threshold=0.3):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs>=threshold)] = 1
    y_true = labels

    f1 = f1_score(y_true, y_pred, average = 'macro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
    hamming = hamming_loss(y_true, y_pred)

    metrics = {
        "roc_auc": roc_auc,
        "hamming_loss": hamming,
        "f1": f1
    }

    return metrics

  def compute_metrics(self,p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    result = self.multi_labels_metrics(predictions=preds,labels=p.label_ids)

    return result

# Custom dataset class for tokenization  
class CustomDataset(torch.utils.data.Dataset):
  def __init__(self, texts, labels, tokenizer, max_len=128):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = str(self.texts[idx])
    label = torch.tensor(self.labels[idx])

    encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')

    return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'labels': label
    }
  

#Tokenization and Model Trainer

class ModelTrainer:
    def __init__(self,config: ModelTrainerConfig):
        self.config = config
        self.multi_label_metric = MultiLabelMetric()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_ckpt)
    

    def tokenize(self):

        dataset_train = pd.read_csv(self.config.data_path_train)
        dataset_validation = pd.read_csv(self.config.data_path_validation)

        # When saving the dataset in transform stage, the labels got converted to strings, so converting them back to floats
        dataset_train['label'] = dataset_train['label'].apply(lambda x: [float(val) for val in x.strip('[]').split(', ')])
        dataset_validation['label'] = dataset_validation['label'].apply(lambda x: [float(val) for val in x.strip('[]').split(', ')])

        train_text = dataset_train['comment_text'].to_list()
        validation_text = dataset_validation['comment_text'].to_list()

        train_labels = dataset_train['label'].to_list()
        validation_labels = dataset_validation['label'].to_list()

        train_tokenized_dataset = CustomDataset(train_text,train_labels,tokenizer=self.tokenizer)
        validation_tokenized_dataset = CustomDataset(validation_text,validation_labels, tokenizer=self.tokenizer)

        return train_tokenized_dataset, validation_tokenized_dataset

    def train(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading the tokenized datasets
        train_dataset_tokenized , validation_dataset_tokenized = self.tokenize()

        # Loading the model
        distilbert_model = AutoModelForSequenceClassification.from_pretrained(self.config.model_ckpt,num_labels=6,problem_type="multi_label_classification").to(device)

        # Loading the training arguments
        training_args = TrainingArguments(
            output_dir = self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit
        )

        #Trainer 
        trainer = Trainer(model=distilbert_model,
                  args=training_args,
                  train_dataset=train_dataset_tokenized,
                  eval_dataset = validation_dataset_tokenized,
                  compute_metrics=self.multi_label_metric.compute_metrics)

        #Model Training
        trainer.train()

        #Saving model and tokenizer
        distilbert_model.save_pretrained(os.path.join(self.config.root_dir,"distilbert_full_finetuning_toxic_comments"))