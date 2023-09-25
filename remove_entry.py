import cv2, gzip, pickle

with gzip.open('dataset.pkl.gz', 'rb') as f:
    dataset = pickle.load(f)

for idx, (img, ans) in enumerate(dataset):
    image = img.reshape(28, 28)
    print(ans)
    cv2.imshow('k', cv2.resize(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), (200, 200)))
    k = cv2.waitKey()
    if k == ord('y'):
        dataset.pop(idx)
    elif k == ord('x'):
        break

with gzip.open('dataset.pkl.gz', 'wb') as f:
    pickle.dump(dataset, f)