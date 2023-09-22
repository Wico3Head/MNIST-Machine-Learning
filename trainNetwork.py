import gzip, pickle, cv2
import numpy as np  
from network import Network
from setting import *

with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

def distortImage(img):
    original_img = cv2.cvtColor((img * 255).reshape((28, 28)), cv2.COLOR_GRAY2BGR)

    scale = np.random.uniform(0.6, 0.8)
    new_width = int(28 * scale)
    new_height = int(28 * scale)
    rescaled_image = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    scaled_img = np.zeros((28, 28, 3), dtype=np.uint8)
    scaled_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = rescaled_image

    angle = np.random.randint(-10, 11)
    img_center = tuple(np.array(scaled_img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated_img = cv2.warpAffine(scaled_img, rot_mat, scaled_img.shape[1::-1], flags=cv2.INTER_LINEAR)

    dx = np.random.randint(-5, 6)
    dy = np.random.randint(-5, 6)
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    translated_img = cv2.warpAffine(rotated_img, translation_matrix, (28, 28))

    result_image = cv2.cvtColor(translated_img, cv2.COLOR_BGR2GRAY)
    return result_image.flatten() / 255

def train_network():
    net = Network([784, 100, 10])
    train_img, train_ans = train_set[0], train_set[1]
    train_set_size = len(train_img)
    for i in range(2):
        epoch_data = []
        for idx in range(train_set_size):
            img = distortImage(train_img[idx])
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