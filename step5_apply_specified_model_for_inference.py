import warnings
warnings.filterwarnings('ignore') 

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate model from local_checkpoint
model_checkpoint = 'initial_model'      # or model_checkpoint = 'distilbert-base-uncased'
# model_checkpoint = 'TL_best_model'
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', add_prefix_space=True)

# model inference for 1 sample
text = 'This is not bad. Although there is still room for improvement in details, it is already above average.'
inputs = tokenizer.encode(text, return_tensors="pt").to(device)     # tokenize text
logits = model(inputs).logits                                       # compute logits
predictions = torch.argmax(logits)                                  # convert logits to label
print("Model prediction:")
print("----------------------------")
print(text + " - " + id2label[predictions.tolist()])

# model inference for a list of samples
text_list = ["It was good.", "Not a fan, don't recommed.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]
print("Model predictions:")
print("----------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text + " - " + id2label[predictions.tolist()])

# model inference for a dataset
dataset = load_dataset('csv', data_files='my-sentiment-test.csv').pop("train")
length = len(dataset)
correct_cnt = 0
for i in tqdm(range(length)):
    inputs = tokenizer.encode(dataset[i]['text'], return_tensors="pt").to(device)
    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    if predictions.tolist() == dataset[i]['label']:
        correct_cnt += 1
print('There are ' + str(correct_cnt) + ' correct predictions.')
print('The accurate rate is ' + str(correct_cnt/length) + '.')