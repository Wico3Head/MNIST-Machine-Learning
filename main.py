import pickle, pygame, sys
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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    canvas = Canvas(screen)

        if drawing:
            mouse = pygame.mouse.get_pos()
            if not canvas.onCanvas(mouse):
                drawing = False
            else:
                canvas.paint(mouse)

        screen.fill(BG_COLOR)
        canvas.draw()
        output = net.activate(canvas.low_res_image.flatten() / 255)
        decision = output.tolist().index(max(output))

        output_label = FONT.render(f"output: {decision}", False, 'white')
        output_rect = output_label.get_rect(topleft=(750, 500))
        screen.blit(output_label, output_rect)

        pygame.display.update()

if __name__ == "__main__":
    main()