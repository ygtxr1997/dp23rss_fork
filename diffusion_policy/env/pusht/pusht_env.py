import gym
import math
import random
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            domain_shift: str = None,
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )
        self.tees = []  # 用于存储所有 T 形物体信息

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action
        self.history_positions = []
        self.first_obs_frame = None

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        specific_state = reset_to_state
        # specific_state = np.array([  # agent,block,block_rot
        #     256+100, 256-100,
        #     256-100, 256+50,
        #     np.pi / 4
        # ])
        self.reset_to_state = specific_state
        self.domain_shift = domain_shift

    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def get_history_positions(self):
        return self.history_positions

    def get_first_obs_frame(self):
        return self.first_obs_frame

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
                self.history_positions.append(self.agent.position)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def random_agent(self):
        RandomAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            old_x, old_y = self.start_x, self.start_y
            dx = random.randint(10 - self.window_size, self.window_size - 10) * 0.3
            dy = random.randint(10 - self.window_size, self.window_size - 10) * 0.3
            if self.start_x + dx >= self.window_size - 10 or self.start_x + dx <= 10:
                self.start_x -= dx
            if self.start_y + dy >= self.window_size - 10 or self.start_y + dy <= 10:
                self.start_y -= dy
            self.start_x += dx
            self.start_y += dy
            act = (self.start_x, self.start_y)
            # mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            # if self.teleop or (mouse_position - self.agent.position).length < 30:
            #     self.teleop = True
            #     act = mouse_position
            return act
        return RandomAgent(act)

    def _get_obs(self):
        # print("[DEBUG] PushTEnv _get_obs() is called.")
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):
        """
        mode: if demo, mode='human'; if eval, mode='rgb_array'
        """
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        ''' baseline: white '''
        canvas.fill((255, 255, 255))
        if self.domain_shift == "orange":
            ''' to orange '''
            # canvas.fill((180, 180, 180))  # to gray, 0.25
            canvas.fill((255, 125, 80))  # to orange, 0.23
        elif self.domain_shift == "texture":
            ''' with texture '''
            image = pygame.image.load("media/blotchy_0015.jpg")  # to texture, 0.17
            for x in range(0, canvas.get_width(), image.get_width()):
                for y in range(0, canvas.get_height(), image.get_height()):
                    canvas.blit(image, (x, y))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        # # 绘制 T 形物体及其自定义图像
        # # self.load_images()
        # for tee in self.tees:
        #     body = tee["body"]
        #     angle = body.angle
        #
        #     # 获取横梁和竖杆的顶点的世界坐标
        #     vertices1_world = [
        #         body.local_to_world((v[0], v[1])) for v in tee["vertices1"]
        #     ]
        #     vertices2_world = [
        #         body.local_to_world((v[0], v[1])) for v in tee["vertices2"]
        #     ]
        #
        #     # 如果图像已加载，使用图像绘制 T 的横梁和竖杆
        #     if tee["image1"] and tee["image2"]:
        #         # 计算横梁的中心点
        #         center1 = (
        #             (vertices1_world[0][0] + vertices1_world[2][0]) / 2,
        #             (vertices1_world[0][1] + vertices1_world[2][1]) / 2,
        #         )
        #         # 绘制横梁图像
        #         image1_rotated = pygame.transform.rotate(tee["image1"], -angle * 57.2958)  # 角度制
        #         rect1 = image1_rotated.get_rect(center=center1)  # 将图像中心对齐到横梁中心
        #         # canvas.blit(image1_rotated, rect1.topleft)
        #
        #         # 计算竖杆的中心点
        #         center2 = (
        #             (vertices2_world[0][0] + vertices2_world[2][0]) / 2,
        #             (vertices2_world[0][1] + vertices2_world[2][1]) / 2,
        #         )
        #         # 绘制竖杆图像
        #         image2_rotated = pygame.transform.rotate(tee["image2"], -angle * 57.2958)  # 角度制
        #         rect2 = image2_rotated.get_rect(center=center2)  # 将图像中心对齐到竖杆中心
        #         # canvas.blit(image2_rotated, rect2.topleft)
        #
        #     # 如果没有加载图像，绘制形状边框（占位）
        #     else:
        #         # 绘制横梁的边框
        #         pygame.draw.polygon(
        #             canvas,
        #             (0, 0, 0),  # 黑色
        #             [(v[0], v[1]) for v in vertices1_world],
        #             width=1,
        #         )
        #         # 绘制竖杆的边框
        #         pygame.draw.polygon(
        #             canvas,
        #             (0, 0, 0),  # 黑色
        #             [(v[0], v[1]) for v in vertices2_world],
        #             width=1,
        #         )

        # # 对于其他非 T 的物体，使用 debug_draw
        # for shape in self.space.shapes:
        #     if shape not in [t["shape1"] for t in self.tees] and shape not in [t["shape2"] for t in self.tees]:
        #         self.space.debug_draw(draw_options)

        # # Draw agent and block.
        # self.space.debug_draw(draw_options)

        ## 添加光照效果, 0.15
        if self.domain_shift == "light":
            self._apply_random_lighting(canvas)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        if self.first_obs_frame is None:
            self.first_obs_frame = img
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            # print(self.render_action, self.latest_action)
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)

                ## Render history actions
                # for act in self.history_actions:
                #     coord = (act / 512 * 96).astype(np.int32)
                #     marker_size = int(8 / 96 * self.render_size)
                #     thickness = int(1 / 96 * self.render_size)
                #     cv2.drawMarker(img, coord,
                #                    color=(255, 255, 0), markerType=cv2.MARKER_CROSS,
                #                    markerSize=marker_size, thickness=thickness)

                # self.history_actions.append(action)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        self.start_x, self.start_y = random.randint(10, self.window_size - 10), random.randint(10, self.window_size - 10)
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        ''' baseline: agent is circle '''
        self.agent = self.add_circle((256, 400), 15)
        ''' agent is a poly '''
        # self.agent = self.add_box_agent((256, 400), 50)

        if self.domain_shift == "size":
            self.block = self.add_tee((256, 300), 0, scale=40)  # baseline: 30
        else:
            self.block = self.add_tee((256, 300), 0, scale=30)  # baseline: 30
        # self.block = self.add_tee((256, 300), 0, scale=30,
        #                           image_path='media/blotchy_0015.jpg')  # image_path='media/blotchy_0015.jpg'
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box_agent(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        w, h = radius, radius
        vs = [(-w/ 2,-h/ 2), (w/ 2,-h/ 2), (w/ 2,h/ 2)]
        shape = pymunk.Poly(body, vs)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS(), image_path=None):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        self.vertices1 = vertices1
        self.vertices2 = vertices2
        self.scale = scale

        # 保存 T 形物体信息，延迟加载图像
        tee = {
            "body": body,
            "shape1": shape1,
            "shape2": shape2,
            "vertices1": vertices1,
            "vertices2": vertices2,
            "scale": scale,
            "image_path": image_path,  # 保存图像路径
            "image1": None,  # 图像占位
            "image2": None  # 图像占位
        }
        # self.tees.append(tee)

        return body

    def load_images(self):
        """加载所有 T 形物体的图像"""
        for tee in self.tees:
            if tee["image_path"]:
                original_image = pygame.image.load(tee["image_path"]).convert_alpha()
                print(f"{tee['image_path']} image loaded.")
                length = 4
                scale = tee["scale"]

                # 缩放横梁和竖杆的图像
                tee["image1"] = pygame.transform.scale(original_image, (int(length * scale), int(scale)))
                tee["image2"] = pygame.transform.scale(original_image, (int(scale), int(length * scale)))

    def _apply_lighting(self, canvas):
        """在场景中添加动态光照效果"""
        # 光源参数
        light_radius = 300  # 光照半径
        light_intensity = 255  # 光强
        light_color = (255, 255, 200)  # 光的颜色

        # 获取光源位置（这里以鼠标为例）
        light_x, light_y = pygame.mouse.get_pos()

        # 创建遮罩层
        darkness = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        darkness.fill((0, 0, 0, 200))  # 黑色半透明遮罩

        # 创建光源
        light_surface = self._create_light_surface(light_radius, light_color, light_intensity)

        # 在遮罩层上叠加光源
        darkness.blit(light_surface, (light_x - light_radius, light_y - light_radius),
                      special_flags=pygame.BLEND_RGBA_SUB)

        # 将遮罩层叠加到场景上
        canvas.blit(darkness, (0, 0))

    def _apply_random_lighting(self, canvas):
        """在场景中添加随机点光源效果"""
        # 光源参数
        light_radius = 250  # 光源半径
        light_intensity = 200  # 光源强度
        light_color = (255, 255, 200)  # 光源颜色
        num_lights = 1  # 随机光源数量

        # 创建遮罩层
        darkness = pygame.Surface(canvas.get_size(), pygame.SRCALPHA)
        darkness.fill((0, 0, 0, 200))  # 黑色半透明遮罩

        # 生成随机光源
        light_x = random.randint(0, self.window_size)
        light_y = random.randint(0, self.window_size)
        for _ in range(num_lights):
            # 随机生成光源位置
            max_delta = int(self.window_size * 0.02)
            light_x += random.randint(-max_delta, max_delta)
            light_y += random.randint(-max_delta, max_delta)

            # 创建光源
            light_surface = self._create_light_surface(light_radius, light_color, light_intensity)

            # 在遮罩层上叠加光源
            darkness.blit(light_surface, (light_x - light_radius, light_y - light_radius),
                          special_flags=pygame.BLEND_RGBA_SUB)

        # 将遮罩层叠加到场景
        canvas.blit(darkness, (0, 0))

    def _create_light_surface(self, radius, color, intensity=255):
        """创建一个渐变的光源图像"""
        light_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for x in range(radius * 2):
            for y in range(radius * 2):
                # 计算当前点到圆心的距离
                distance_to_center = math.sqrt((x - radius) ** 2 + (y - radius) ** 2)
                if distance_to_center <= radius:
                    # 计算透明度 (Alpha)，随着距离递减
                    alpha = max(0, intensity - (distance_to_center / radius) * intensity)
                    light_surface.set_at((x, y), (*color, int(alpha)))
        return light_surface

