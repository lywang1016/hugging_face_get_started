from datasets import load_dataset

# Load dataset from hugging face
raw_dataset = load_dataset('stanfordnlp/sst2')
print(raw_dataset)

# Save the dataset to local CSV file
for split, dataset in raw_dataset.items():
    dataset.to_csv(f'sst2-sentiment-{split}.csv', index=None)