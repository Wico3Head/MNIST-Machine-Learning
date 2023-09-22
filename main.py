import gzip, pickle, pygame, cv2, sys
import numpy as np    
from network import Network
from canvas import Canvas
from setting import *
pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Handwritten Digit Recogniser')

def main():
    with open('network.pkl', 'rb') as f:
        net: Network = pickle.load(f)

    canvas = Canvas(screen)
    drawing = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if canvas.onCanvas(event.pos):
                    drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                if drawing:
                    drawing = False

        if drawing:
            mouse = pygame.mouse.get_pos()
            if not canvas.onCanvas(mouse):
                drawing = False
            else:
                canvas.paint(mouse)

        screen.fill(BG_COLOR)
        canvas.draw()
        output = net.activate(canvas.low_rest_image.flatten())
        #print(drawing)
        pygame.display.update()

if __name__ == "__main__":
    main()