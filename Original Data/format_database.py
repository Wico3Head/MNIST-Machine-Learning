import gzip, pickle
import numpy as np

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_img, train_ans = train_set[0], train_set[1]
valid_img, valid_ans = valid_set[0], valid_set[1]
test_img, test_ans = test_set[0], test_set[1]

imgs = []
ans = []
for i in range(len(train_img)):
    imgs.append(train_img[i])
    ans.append(train_ans[i])

for i in range(len(valid_img)):
    imgs.append(valid_img[i])
    ans.append(valid_ans[i])

for i in range(len(test_img)):
    imgs.append(test_img[i])
    ans.append(test_ans[i])

dataset = [(imgs[i], ans[i]) for i in range(len(imgs))]
print(dataset[0])
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)