from datasets import load_dataset, concatenate_datasets, DatasetDict
from icecream import ic

data_files = {
    'train': 'sst2-sentiment-train.csv',
    'validation': 'sst2-sentiment-validation.csv',
    'test': 'sst2-sentiment-test.csv',
}

raw_dataset_train = load_dataset('csv', data_files=data_files, split='train')
raw_dataset_validation = load_dataset('csv', data_files=data_files, split='validation')

raw_dataset = concatenate_datasets([raw_dataset_train, raw_dataset_validation])
ic('raw_dataset')
print(raw_dataset)

clean_dataset = raw_dataset.remove_columns(['idx'])
ic('clean_dataset')
print(clean_dataset)

shuffled_dataset = clean_dataset.shuffle(seed=666) # actually train_test_split function also do shuffle

renamed_dataset = shuffled_dataset.rename_column('sentence', 'text')
ic('renamed_dataset')
print(renamed_dataset)

temp_dataset1 = renamed_dataset.train_test_split(test_size=0.04)
ic('temp_dataset1')
print(temp_dataset1)

train_dataset = temp_dataset1.pop('train')
ic('train_dataset')
print(train_dataset)

test_dataset_temp = temp_dataset1.pop('test')
temp_dataset2 = test_dataset_temp.train_test_split(test_size=0.5)
# print(temp_dataset2)

test_dataset = temp_dataset2.pop('train')
validation_dataset = temp_dataset2.pop('test')
ic('validation_dataset')
print(validation_dataset)
ic('test_dataset')
print(test_dataset)

my_dataset = DatasetDict({'train':train_dataset,
                          'validation':validation_dataset,
                          'test':test_dataset})
ic('my_dataset')
print(my_dataset)

for split, dataset in my_dataset.items():
    dataset.to_csv(f'my-sentiment-{split}.csv', index=None)