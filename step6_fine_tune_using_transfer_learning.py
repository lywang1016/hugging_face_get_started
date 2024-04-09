import warnings
warnings.filterwarnings('ignore') 

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

import evaluate
import numpy as np
import json

# load training dataset to evaluate the model
data_files = {
    'train': 'my-sentiment-train.csv',
    'validation': 'my-sentiment-validation.csv',
}
dataset = load_dataset('csv', data_files=data_files)

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate model from local_checkpoint
model_checkpoint = 'initial_model'
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# create tokenizer
with open(model_checkpoint+'/config.json') as json_file:
    config = json.load(json_file)
tokenizer = AutoTokenizer.from_pretrained(config["_name_or_path"], add_prefix_space=True)

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]
    # tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

# define training arguments
training_args = TrainingArguments(
    output_dir="TL-auto-save",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

# save locally
model.save_pretrained("TL_best_model")