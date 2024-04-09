import warnings
warnings.filterwarnings('ignore') 

from transformers import AutoModelForSequenceClassification

# indicate which model you want from hugging face
model_checkpoint = 'distilbert-base-uncased'

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# download the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# save the model to local
model.save_pretrained("initial_model")