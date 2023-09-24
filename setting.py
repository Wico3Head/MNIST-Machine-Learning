import pygame, os
import numpy as np
pygame.init()

LEARN_RATE = 0.42
REGULARISATION_CONSTANT = 12
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

BG_COLOR = (51, 51, 51)
RETRAINING_BG_COLOR = (255, 255, 255, 190)
ADD_TO_DATABASE_MENU_COLOR = (70, 70, 70)
LOCAL_DIR = os.path.dirname(__file__)
FONT = pygame.font.Font(os.path.join(LOCAL_DIR, 'Font/pixel.ttf'), 50)
SMALL_FONT = pygame.font.Font(os.path.join(LOCAL_DIR, 'Font/pixel.ttf'), 40)

PEN_RADIUS = 25