import gzip, pickle
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
with gzip.open('dataset.pkl.gz', 'wb') as f:
    pickle.dump(dataset, f)