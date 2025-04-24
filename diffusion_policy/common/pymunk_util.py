import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from PIL import Image, ImageDraw
import random
import math


COLLTYPE_DEFAULT = 0
COLLTYPE_MOUSE = 1
COLLTYPE_BALL = 2

def get_body_type(static=False):
    body_type = pymunk.Body.DYNAMIC
    if static:
        body_type = pymunk.Body.STATIC
    return body_type


def create_rectangle(space,
        pos_x,pos_y,width,height,
        density=3,static=False):
    body = pymunk.Body(body_type=get_body_type(static))
    body.position = (pos_x,pos_y)
    shape = pymunk.Poly.create_box(body,(width,height))
    shape.density = density
    space.add(body,shape)
    return body, shape


def create_rectangle_bb(space, 
        left, bottom, right, top, 
        **kwargs):
    pos_x = (left + right) / 2
    pos_y = (top + bottom) / 2
    height = top - bottom
    width = right - left
    return create_rectangle(space, pos_x, pos_y, width, height, **kwargs)

def create_circle(space, pos_x, pos_y, radius, density=3, static=False):
    body = pymunk.Body(body_type=get_body_type(static))
    body.position = (pos_x, pos_y)
    shape = pymunk.Circle(body, radius=radius)
    shape.density = density
    shape.collision_type = COLLTYPE_BALL
    space.add(body, shape)
    return body, shape

def get_body_state(body):
    state = np.zeros(6, dtype=np.float32)
    state[:2] = body.position
    state[2] = body.angle
    state[3:5] = body.velocity
    state[5] = body.angular_velocity
    return state


class ImageLightingEffect:
    def __init__(self, light_radius=50, light_intensity=200, light_color=(255, 255, 200), num_lights=1):
        self.light_radius = light_radius
        self.light_intensity = light_intensity
        self.light_color = light_color
        self.num_lights = num_lights
        self.max_width = 96
        self.max_height = 96

        self.light_x = random.randint(0, self.max_width)
        self.light_y = random.randint(0, self.max_height)

    def apply_random_lighting(self, image: Image.Image):
        """在图像上添加随机点光源效果"""
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # 创建遮罩层

        for _ in range(self.num_lights):
            # 随机生成光源位置
            max_delta = int(self.max_width * 0.02)
            self.light_x += random.randint(-max_delta, max_delta)
            self.light_y += random.randint(-max_delta, max_delta)

            # 创建光源并叠加到遮罩层
            light_surface = self._create_light_surface()
            overlay.paste(light_surface,
                          (self.light_x - self.light_radius, self.light_y - self.light_radius),
                          light_surface)

        # 将遮罩层叠加到原图
        image = Image.alpha_composite(image.convert("RGBA"), overlay)
        return image

    def _create_light_surface(self):
        """创建一个渐变的光源图像"""
        diameter = self.light_radius * 2
        light_surface = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        draw = ImageDraw.Draw(light_surface)

        for r in range(self.light_radius, 0, -1):
            alpha = int(self.light_intensity * (r / self.light_radius))
            color = (*self.light_color, alpha)
            draw.ellipse(
                (self.light_radius - r, self.light_radius - r, self.light_radius + r, self.light_radius + r),
                fill=color
            )

        return light_surface

