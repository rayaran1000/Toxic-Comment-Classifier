
# Multi Label Toxic Comment classification

![image](https://miro.medium.com/v2/resize:fit:720/format:webp/1*PIs25RW-zFYGalzlzgdI9A.jpeg)


This project aims to implement multi label text classification using the Distilbert model from Hugging Face, fine-tuned on the Toxic Comment Jigsaw dataset. Text classification is the process of classifying a piece of text into a set of target labels. The Distilbert model, based on transformer architecture, has shown promising results in various natural language processing tasks, including text classification.

Distilbert is a distilled version of BERT Model. Distillation(compression of the model) consists of the model weights being decreased, while maintaining almost similar performance as the big model

By fine-tuning Distilbert on Toxic Comment Dataset, which consists of multi label classification of internet comments, we aim to create a model that can accurately classify whether a comment is toxic or not. If the comment is toxic, the type of toxicity is also included as part of the classification. The project involves data preprocessing, model fine-tuning, and potentially deployment for real-world applications.


## Directory Structure

```plaintext
/project
│   README.md
│   requirements.txt
|   application.py
|   setup.py
|   template.py
|   Dockerfile
|   params.yaml
└───.github/workflows
|   └───main.yaml
└───research
|   └───Toxic Comment Analysis using BERT.ipynb
|   └───01_data_ingestion.ipynb
|   └───02_data_validation.ipynb
|   └───03_data_transformation.ipynb
|   └───04_model_trainer.ipynb 
|   └───trials.ipynb 
└───src/ToxicCommentClassifier
|   └───components
|       └───data_ingestion.py
|       └───data_transformation.py
|       └───data_validation.py
|       └───model_trainer.py
|   └───config
|       └───configuration.py
|   └───config
|       └───configuration.py
|   └───constants
|   └───entity
|   └───logging
|   └───utils
|   └───pipeline
|       └───prediction.py
|       └───stage_01_data_ingestion.py
|       └───stage_02_data_validation.py
|       └───stage_03_data_transformation.py
|       └───stage_04_model_trainer.py

```

# Installation
### STEPS:

Clone the repository

```bash
https://github.com/rayaran1000/Toxic-Comment-Classifier
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n toxic python=3.8 -y
```

```bash
conda activate toxic
```


### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Finally run the following command
```bash
python app.py
```

Now,
```bash
open up you local host and port 

URL -> localhost:8080
```


```bash
Author: Aranya Ray
Data Scientist
Email: aranya.ray1998@gmail.com

```




    
# Deployment

1. Created IAM user for deployment.

![IAM User creation](https://github.com/rayaran1000/Toxic-Comment-Classifier/assets/122597408/443b4b2d-bce2-45c7-8b12-5344a8236a81)
2. Created ECR repo to store/save docker image.

![ECR Repo](https://github.com/rayaran1000/Toxic-Comment-Classifier/assets/122597408/de1e2441-8f88-4d87-a46a-e854fbac948d)

3. Created EC2 instance (Ubuntu) and installed docker in EC2 Machine.
   
![EC2 Instance](https://github.com/rayaran1000/Toxic-Comment-Classifier/assets/122597408/0c51549f-ec00-49e9-a056-0bd1d46e727f)

4. Setup Github secrets completed.

![Github actions](https://github.com/rayaran1000/Toxic-Comment-Classifier/assets/122597408/7be05650-8438-468d-9c66-d64064927d23)

## Dataset and Model Specifications

### Dataset 
Toxic Comment Jigsaw dataset : Arsive/toxicity_classification_jigsaw

Documentation : https://huggingface.co/datasets/Arsive/toxicity_classification_jigsaw

### Model
Model used -> distilbert/distilbert-base-uncased

Documentation : https://huggingface.co/distilbert/distilbert-base-uncased
## Acknowledgements

I gratefully acknowledge the contributions of the developers and researchers behind the Hugging Face Transformers library, which provides access to state-of-the-art NLP models like Distilbert. 

Additionally, we extend our appreciation to the creators of the Toxic Comment Jigsaw dataset, whose efforts in compiling and curating the dataset have been invaluable to our research. 

Special thanks to the open-source community for their continuous support and valuable feedback.

