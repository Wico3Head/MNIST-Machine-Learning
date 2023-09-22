import gzip, pickle, pygame, cv2, sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt     
from network import Network

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

print(len(train_set[0]))