from datasets import load_dataset

dataset_for_test = load_dataset('csv', data_files='my-sentiment-test.csv').pop('train')
print(dataset_for_test)

idx = 17

print(dataset_for_test[idx]['text'])
print(dataset_for_test[idx]['label'])