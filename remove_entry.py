import pickle, cv2, gzip
with gzip.open('dataset.pkl.gz', 'rb') as file:
    dataset = pickle.load(file, encoding='latin1')

for idx, (img, ans) in enumerate(dataset[::-1]):
    image = cv2.cvtColor(img.reshape((28, 28)), cv2.COLOR_GRAY2BGR)
    print(ans)
    cv2.imshow('l', cv2.resize(image, (500, 500)))
    k = cv2.waitKey()
    if k == ord('y'):
        dataset.pop()[idx]
    elif k == ord('x'):
        break

with gzip.open('dataset.pkl.gz', 'wb') as f:
    pickle.dump(dataset, f)