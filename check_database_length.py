import gzip, pickle
with gzip.open('dataset.pkl.gz', 'rb') as f:
    dataset = pickle.load(f)
print(len(dataset))