#Prediction Pipeline
from ToxicCommentClassifier.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_prediction_pipeline_config() # Used to extract finetuned model path and tokenizer configuration

    def predict(self,text):

        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_ckpt)
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.config.finetuned_model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        finetuned_model.to(device)

        encoding = tokenizer(text,return_tensors='pt')

        encoding = {key: tensor.to(device) for key, tensor in encoding.items()}

        outputs = finetuned_model(**encoding)

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(outputs.logits[0].cpu()) # This part needs to be present in CPU
        preds = np.zeros(probs.shape)
        preds[np.where(probs>=0.3)] = 1 # 0.3 is the threshold value choosen during evaluation metrics
        prediction = pd.DataFrame(preds,label_columns,columns=['Prediction'])
        return prediction
