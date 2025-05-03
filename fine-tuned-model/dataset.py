from datasets import load_dataset

# this will download and cache the data locally
dataset = load_dataset("bigbio/n2c2_2014_deid")

# inspect splits
print(dataset)       # e.g. {'train': Dataset, 'test': Dataset}
print(dataset['train'][0])  # view the first example
