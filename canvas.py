import pygame, cv2, math
import numpy as np
from setting import *

class Canvas:
    def __init__(self, screen):
        self.width = 600
        self.height = 600
        self.pos = (50, 50)
        self.screen = screen
        self.image = np.zeros((self.width, self.height), dtype=np.uint8)
        self.bg = pygame.Rect(self.pos[0], self.pos[1], self.width, self.height)
        self.low_res_image = self.convertToLowRes()
        self.pixel_size = 600 / 28

    def convertToLowRes(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        new_res = (28, 28)
        resized_image = cv2.resize(image, new_res)
        low_res_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        return low_res_image
    
    def onCanvas(self, pos):
        return self.pos[0] < pos[0] <= self.pos[0] + self.width and self.pos[1] < pos[1] <= self.pos[1] + self.height
    
    def paint(self, pos):
        canvas_pos_x = pos[0] - self.pos[0]
        canvas_pos_y = pos[1] - self.pos[1]
        replacement_shape = self.image[canvas_pos_y - PEN_RADIUS:canvas_pos_y + PEN_RADIUS, canvas_pos_x - PEN_RADIUS:canvas_pos_x + PEN_RADIUS].shape
        self.image[canvas_pos_y - PEN_RADIUS:canvas_pos_y + PEN_RADIUS, canvas_pos_x - PEN_RADIUS:canvas_pos_x + PEN_RADIUS] = np.full(replacement_shape, 255, dtype=np.uint8)
        self.low_res_image = self.convertToLowRes()
    
    def draw(self):
        pygame.draw.rect(self.screen, 'black', self.bg)
        for y in range(28):
            for x in range(28):
                pixel_greyscale_value = self.low_res_image[y][x]
                if pixel_greyscale_value != 0:
                    color = [pixel_greyscale_value for i in range(3)]
                    rect = pygame.Rect(self.pos[0] + x * self.pixel_size,
                                       self.pos[1] + y * self.pixel_size,
                                       math.ceil(self.pixel_size),
                                       math.ceil(self.pixel_size))
                    pygame.draw.rect(self.screen, color, rect)