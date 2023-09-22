import gzip, pickle, pygame, cv2, sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt     
from network import Network

LEARN_RATE = 0.35
REGULARISATION_CONSTANT = 10

def main():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    net = Network([784, 100, 10])
    train_img, train_ans = train_set[0], train_set[1]
    train_set_size = len(train_img)
    epoch_data = []
    for idx in range(train_set_size):
        img = train_img[idx]
        expected_output = np.array([0 if train_ans[idx] != i else 1 for i in range(10)])
        epoch_data.append([img, expected_output])
        if len(epoch_data) == 50:
            net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)
            epoch_data.clear()
    if epoch_data:
        net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)

    test_img, test_ans = test_set[0], test_set[1]
    test_set_size = len(test_set[0])
    correct = 0
    for idx in range(test_set_size):    
        outputs = net.activate(test_img[idx].flatten())
        decision = outputs.tolist().index(max(outputs))
        if decision == test_ans[idx]:
            correct += 1
            
    print(correct/len(test_ans)*100)

if __name__ == "__main__":
    main()