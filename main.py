import pygame, sys, time, math
from network import Network
import numpy as np
pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
LEARN_RATE = 0.5
REGULARISATION_CONSTANT = 2
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Machine Learning Demo')

def testInput(input):
    result = 1 if input[0] < input[1] else 0
    return np.array([1 - result, result])

def main():
    net = Network([2, 3, 2])
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        training_data = []
        for i in range(300):
            x = np.random.rand()
            y = np.random.rand()
            inputs = [x, y]
            testInputs = [x * 100, y * 100]
            training_data.append(np.array([inputs, testInput(testInputs)]))
        net.learn(training_data, LEARN_RATE, REGULARISATION_CONSTANT)

        screen.fill('white')
        for x in range(100):
            for y in range(100):
                outputs = net.activate(np.array([x/100, y/100]))
                if outputs[0] > outputs[1]:
                    pygame.draw.rect(screen, 'black', pygame.Rect(x * 8, y * 8, 8, 8))

        pygame.display.update()

def main2():
    net = Network([1, 1])
    net.weights[0][0] = -5
    net.bias[0][0] = 0
    while True:  
        print(net.activate([1]))
        net.learn(np.array([[[1], [0]]]), 1.5)      
        time.sleep(0.1)

if __name__ == "__main__":
    main()
    