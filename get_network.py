import gzip, pickle
import numpy as np  
from network import Network
from setting import *

with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

def train_network():
    net = Network([784, 100, 10])
    train_img, train_ans = train_set[0], train_set[1]
    train_set_size = len(train_img)
    epoch_data = []
    for idx in range(train_set_size):
        img = train_img[idx]
        expected_output = np.array([0 if train_ans[idx] != i else 1 for i in range(10)])
        epoch_data.append([img, expected_output])
        if len(epoch_data) == 30:
            net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)
            epoch_data.clear()
    if epoch_data:
        net.learn(epoch_data, LEARN_RATE, REGULARISATION_CONSTANT, train_set_size)

    with open('network.pkl', 'wb') as f:
        pickle.dump(net, f)

if __name__ == "__main__":
    train_network()