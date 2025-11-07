import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
import matplotlib.pyplot as plt
import pydantic
import requests
import base64
import io
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class StepRequestFromEvaluator(pydantic.BaseModel):
    """
    Sent from evaluator to policy.

        gt_video (B,Ts,H,W,3) uint8 \n
        tcp_state: (B,Ts,D) float32, optional \n
        stage_flag: 0:cold start, 1:hot start \n
        instruction: str
    """
    instruction: str
    stage_flag: int  # 0:cold start, 1:hot start

    gt_video: List[List[str]]  # len1=B, len2=Ts (Ts>=v1), each str is base64 of (H,W,rgb)
    tcp_state: Optional[List[List[List[float]]]] = None # len1=B, len2=Ts, len3=D, float32

    def decode_to_raw(self) -> Dict[str, Any]:
        def base64_to_image_H_W_C(img_base64: str) -> np.ndarray:
            img_byte = base64.b64decode(img_base64)
            img_pil = Image.open(io.BytesIO(img_byte), formats=["JPEG"])
            img_np = np.array(img_pil)  # (H,W,3) uint8
            return img_np

        def base64_to_video_T_H_W_C(vid_base64: List[str]) -> np.ndarray:
            return np.stack([base64_to_image_H_W_C(img_b64) for img_b64 in vid_base64], axis=0)

        gt_video_B_T_H_W_C = np.stack([base64_to_video_T_H_W_C(vid_b64) for vid_b64 in self.gt_video], axis=0)  # (B,Ts,H,W,3) uint8
        tcp_state_B_T_D = None if self.tcp_state is None else np.array(self.tcp_state, dtype=np.float32)  # (B,Ts,2) float32
        return {
            "instruction": self.instruction,
            "stage_flag": self.stage_flag,
            "gt_video": gt_video_B_T_H_W_C,
            "tcp_state": tcp_state_B_T_D,
        }

    @classmethod
    def encode_from_raw(cls,
                        instruction: str,
                        stage_flag: int,
                        gt_video: np.ndarray,  # (B,Ts,H,W,3) uint8
                        tcp_state: Optional[np.ndarray] = None,  # (B,2) float32
                        ) -> 'StepRequestFromEvaluator':
        instance = cls(
            instruction=instruction,
            stage_flag=stage_flag,
            gt_video=cls.video_np_to_base64(gt_video),
            tcp_state=None if tcp_state is None else tcp_state.tolist()
        )
        return instance

    @staticmethod
    def video_np_to_base64(video: np.ndarray) -> List[List[str]]:
        assert video.ndim in [4, 5], f"video should be (B,Ts,H,W,3) or (Ts,H,W,3), got {video.shape}"
        assert video.dtype == np.uint8

        def image_to_base64(img_H_W_C) -> str:
            image_pil = Image.fromarray(img_H_W_C)
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format="JPEG")
            image_bytes = image_bytes.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            return image_base64

        def video_to_base64(vid_T_H_W_C) -> List[str]:
            return [image_to_base64(img) for img in vid_T_H_W_C]

        if video.ndim == 4:  # No batch-dim
            video_base64 = [video_to_base64(video)]  # add batch-dim
        elif video.ndim == 5:
            video_base64 = [video_to_base64(vid) for vid in video]
        else:
            raise NotImplementedError

        return video_base64


class StepRequestFromPolicy(pydantic.BaseModel):
    """
    Sent from policy to evaluator.

        action: (B,v2,D) float32 \n
        max_cache_action: int, optional, tell evaluator how many actions to cache, None means
    """

    action: List[List[List[float]]]  # len1=B, len2=v2 (v2=H), len3=D
    max_cache_action: int = None  # tell evaluator how many actions to cache, None means no limit

    def decode_to_raw(self) -> Dict[str, Any]:
        return {
            "action": np.array(self.action, dtype=np.float32)  # (B,v2,D) float32
        }

    @classmethod
    def encode_from_raw(cls, action: np.ndarray) -> 'StepRequestFromPolicy':
        if action.ndim == 2:  # (H,D)
            action = action[np.newaxis, ...]  # add batch dim
        assert action.ndim == 3, f"action should be (B,H,D), got {action.shape}"
        assert action.dtype in [np.float32, np.float64], f"action should be float32 or float64, got {action.dtype}"
        instance = cls(action=action.tolist())
        return instance


class PushTImageSocketRunner(BaseImageRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 n_train_vis=3,
                 train_start_seed=0,
                 n_test=22,
                 n_test_vis=6,
                 legacy_test=False,
                 test_start_seed=10000,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=96,  # ori:96
                 past_action=False,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 reset_to_state=None,
                 domain_shift=None,
                 save_name=None,
                 # socket related
                 policy_url='localhost:6006',
                 send_per_frames: int = 1,
                 ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size,
                        reset_to_state=reset_to_state,
                        domain_shift=domain_shift,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.reset_to_state = reset_to_state
        self.domain_shift = domain_shift
        self.save_name = save_name

        # socket related
        self.policy_url = policy_url
        self.send_per_frames = send_per_frames
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.http_session = requests.Session()
        self.task_instruction = None
        self.cache_frames_B_H_W_C = None

    def init_socket(self, task_instruction: str) -> int:
        resp = self.http_session.get(f"{self.policy_url}/init")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.send_per_frames = max(max_cache_action, 1)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.cache_frames_B_H_W_C = None

        self.task_instruction = task_instruction
        print("[PushTImageSocketRunner] Connected to policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames}",
              f"task_instruction len={len(task_instruction)}")
        return max_cache_action

    def send_reset(self, task_instruction: str = None) -> int:
        assert self.task_instruction is not None, "Please call init_socket first."
        self.task_instruction = task_instruction if task_instruction is not None else self.task_instruction

        resp = self.http_session.get(f"{self.policy_url}/reset")
        resp.raise_for_status()
        resp = resp.json()

        max_cache_action = resp["max_cache_action"]
        self.send_per_frames = max(max_cache_action, 1)
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.cache_frames_B_H_W_C = None

        print("[PushTImageSocketRunner] Reset policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames},"
              f" task_instruction len={len(self.task_instruction)}")
        return max_cache_action

    def send_obs_and_get_action(self,
                                image_B_T_H_W_C: np.ndarray,
                                stage_flag: int,
                                robot_states_B_T_D: np.ndarray = None,
                                ) -> np.ndarray:
        assert self.task_instruction is not None, "Please call init_socket first."

        if self.send_cnt % self.send_per_frames == 0:
            video_B_v2_H_W_C = image_B_T_H_W_C
            request_to_policy = StepRequestFromEvaluator.encode_from_raw(
                instruction=self.task_instruction,
                stage_flag=stage_flag,
                gt_video=video_B_v2_H_W_C,  # (B,Ts,H,W,3) uint8
                tcp_state=robot_states_B_T_D,  # (B,Ts,D) float32, not used
            )

            # print(f"[PushTImageSocketRunner] sending frames, i={self.send_cnt}, per={self.send_per_frames}, "
            #       f"i%per={self.send_cnt % self.send_per_frames}, video_B_v2_H_W_C:{video_B_v2_H_W_C.shape}, ")
            sending_dict = request_to_policy.model_dump(mode="json")
            response = self.http_session.post(
                f"{self.policy_url}/step",
                json=sending_dict
            )
            response.raise_for_status()

            response = response.json()
            raw_actions = StepRequestFromPolicy(action=response["action"]).decode_to_raw()["action"]  # (B,H,7)

            # assert raw_actions.shape[0] == self.send_per_frames
            self.cache_actions_B_T_D = raw_actions
            self.cache_frames_B_H_W_C = None
        else:  # don't send anything, to save network bandwidth
            pass

        ret_actions = self.cache_actions_B_T_D[:, self.send_cnt % self.send_per_frames]  # (B,D)
        ret_actions_B_1_D = ret_actions[:, None]  # [B,D] -> (B,1,D)

        self.send_cnt += 1
        return ret_actions_B_1_D

    def run(self, device: Union[torch.device, str] = "cuda", close_online: bool = True):
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        first_obs_img = None
        if self.reset_to_state is not None:
            plt.figure(figsize=(8, 8))

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                          args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            # policy.reset()
            stage_flag = 0  # 0:cold start, 1:hot start

            pbar = tqdm.tqdm(total=self.max_steps,
                             desc=f"Eval PushtImageRunner shift={self.domain_shift} {chunk_idx + 1}/{n_chunks}",
                             leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            step_idx = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :, -(self.n_obs_steps - 1):].astype(np.float32)
                if first_obs_img is None:
                    first_obs_img = env.call("get_first_obs_frame")[0]  # for visualization

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # from robokit.debug_utils.printer import print_batch
                # print_batch("[DEBUG] obs_dict", obs_dict)
                '''
                [DEBUG] obs_dict: Dict,keys=dict_keys(['agent_pos', 'image'])
                agent_pos,<class 'torch.Tensor'>,shape=torch.Size([25, 2, 2]), in [0,512]
                image,<class 'torch.Tensor'>,shape=torch.Size([25, 2, 3, 256, 256]), in [0,1]
                '''

                latest_obs_B_H_W_C = (obs_dict["image"][:, -1].permute(0, 2, 3, 1).cpu().numpy() * 255.).astype(np.uint8)
                latest_obs_B_T_H_W_C = (obs_dict["image"].permute(0, 1, 3, 4, 2).cpu().numpy() * 255.).astype(np.uint8)

                latest_robot_states_B_T_D = obs_dict["agent_pos"].cpu().numpy()

                # run policy
                with torch.no_grad():
                    # action_dict = policy.predict_action(obs_dict)
                    # action_dict = {
                    #     "action": torch.randn((len(this_init_fns), 12, 2)).to(device=device) * 20 + 256.,
                    # }
                    out_action_B_1_D = self.send_obs_and_get_action(
                        image_B_T_H_W_C=latest_obs_B_T_H_W_C,  # (B,V*T,H,W,C) uint8
                        stage_flag=stage_flag,
                        robot_states_B_T_D=latest_robot_states_B_T_D,  # (B,T,2) float32
                    )
                    action_dict = {
                        "action": torch.from_numpy(out_action_B_1_D).to(device=device),  # (B,1,2)
                    }

                    if close_online:
                        pass
                    else:
                        stage_flag = 1  # NOTE: after first step, all are hot start

                # print("[DEBUG] action:", action_dict["action"].min(), action_dict["action"].max(), action_dict["action"].shape,
                #       "agent_pos:", obs_dict["agent_pos"].min(), obs_dict["agent_pos"].max(), obs_dict["agent_pos"].shape,
                #       "image:", obs_dict["image"].min(), obs_dict["image"].max(), obs_dict["image"].shape)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)  # (n_envs,n_action_steps,D)
                done = np.all(done)
                past_action = action

                # debug
                debug_rewards = env.call('get_attr', 'reward')[this_local_slice]
                # consider that length of each env_reward may differ
                debug_max_values = []
                for env_reward in debug_rewards:
                    if len(env_reward) > 0:
                        debug_max_values.append(np.max(env_reward))
                    else:
                        debug_max_values.append(0.0)  # 如果为空，设为0
                debug_max_values = np.array(debug_max_values)  # (n_envs,)
                debug_mean_of_max_values = np.mean(debug_max_values)
                debug_max_values_print = ", ".join([f"{x:.4f}" for x in debug_max_values])
                if step_idx % 150 == 148:
                # if step_idx % 20 == 0:
                    print("[DEBUG] IoU max_values:", debug_max_values_print,
                          f"mean_of_max_values: {debug_mean_of_max_values:.4f}", )

                # update pbar
                step_idx += 1
                pbar.update(action.shape[1])
            pbar.close()

            if self.reset_to_state is not None:
                # Save trajectories (positions)
                plt.imshow(np.flipud(first_obs_img), origin='lower')

                history_pos = env.call_each('get_history_positions')
                max_len = 0
                for h in history_pos:
                    max_len = max(max_len, len(h))
                all_positions = []
                for env_idx in range(len(history_pos)):
                    positions = history_pos[env_idx]
                    # print(env_idx, len(positions))
                    np_positions = []
                    for pos in positions:
                        x = float(pos[0])
                        y = float(pos[1])
                        np_positions.append((x, y))
                    np_positions = np.array(np_positions)

                    # 定义卡尔曼滤波器参数
                    initial_state = np.array([0, 0])
                    initial_covariance = np.eye(2) * 1
                    transition_matrix = np.eye(2)
                    observation_matrix = np.eye(2)
                    process_covariance = np.eye(2) * 0.01
                    observation_covariance = np.eye(2) * 100

                    np_positions = moving_average_smooth(np_positions, window_size=40)
                    log_positions = np.pad(np_positions, ((0, max_len - len(np_positions)), (0, 0)), mode='constant')
                    all_positions.append(log_positions[None, :])

                    # np_positions = kalman_filter(
                    #     np_positions, initial_state, initial_covariance,
                    #     transition_matrix, observation_matrix,
                    #     process_covariance, observation_covariance)

                    x, y = np_positions[:, 0], np_positions[:, 1]
                    y = 512 - y  # flip across y=512/2

                    # 绘制运动轨迹

                    # plt.plot(x, y, marker="o", linestyle="-", color="blue", markersize=4, label="运动轨迹")
                    # 颜色和透明度都随着时间变化
                    colors = np.linspace(0, 1, max_len)[:len(x)]  # 颜色从 0 到 1 的渐变值
                    alphas_first = np.linspace(1, 0.1, 1000)
                    alphas_second = np.linspace(0.1, 0.001, max_len - 1000)
                    alphas = np.concatenate((alphas_first, alphas_second), axis=0)[:len(x)]
                    # alphas = np.linspace(1, 0.02, max_len)[:len(x)]  # 透明度从 1 到 0.2

                    # 将颜色和透明度结合，生成 RGBA 格式
                    rgba_colors = plt.cm.viridis(colors)  # 获取渐变颜色
                    rgba_colors[:, 3] = alphas  # 修改透明度
                    plt.scatter(x, y, color=rgba_colors, s=1)

            if self.reset_to_state is not None:
                plt.xlim(0, 512)
                plt.ylim(0, 512)
                # plt.xlabel("X 坐标")
                # plt.ylabel("Y 坐标")
                # plt.title("运动轨迹绘制")
                # plt.grid(True)
                # plt.legend()
                plt.axis('off')

                # 保存图像
                save_name = self.save_name
                plt.savefig(f"data/pusht_eval_output/{save_name}.png", bbox_inches='tight', pad_inches=0)
                all_positions = np.concatenate(all_positions, axis=0)
                print("all_positions:", all_positions.shape, "saved to:", save_name)
                np.save(f"data/pusht_eval_output/{save_name}.npy", all_positions)

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


def moving_average_smooth(trajectory, window_size):
    smoothed_trajectory = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        if i < window_size:
            cur_win = i
        else:
            cur_win = window_size
        start = max(0, i - cur_win)
        end = min(len(trajectory), i + cur_win + 1)
        smoothed_trajectory[i] = np.mean(trajectory[start:end], axis=0)
    return smoothed_trajectory


def kalman_filter(observations, initial_state, initial_covariance, transition_matrix, observation_matrix,
                  process_covariance, observation_covariance):
    """
    卡尔曼滤波器，初始点位置保持不变
    :param observations: 观测值，形状为 (N, 2)
    :param initial_state: 初始状态估计 (2,)
    :param initial_covariance: 初始协方差矩阵 (2, 2)
    :param transition_matrix: 状态转移矩阵 (2, 2)
    :param observation_matrix: 观测矩阵 (2, 2)
    :param process_covariance: 过程噪声协方差 (2, 2)
    :param observation_covariance: 观测噪声协方差 (2, 2)
    :return: 滤波后的状态估计
    """
    n_timesteps = observations.shape[0]
    n_state_vars = initial_state.shape[0]

    # 初始化
    filtered_states = np.zeros((n_timesteps, n_state_vars))
    keep_same = 0
    filtered_states[:keep_same] = observations[:keep_same]  # 初始点保持不变
    state = initial_state
    covariance = initial_covariance

    for t in range(keep_same, n_timesteps):  # 从第 1 个点开始滤波
        # 预测阶段
        predicted_state = np.dot(transition_matrix, state)
        predicted_covariance = np.dot(transition_matrix,
                                      np.dot(covariance, transition_matrix.T)) + process_covariance

        # 更新阶段
        observation = observations[t]
        innovation = observation - np.dot(observation_matrix, predicted_state)
        innovation_covariance = np.dot(observation_matrix,
                                       np.dot(predicted_covariance, observation_matrix.T)) + observation_covariance
        kalman_gain = np.dot(predicted_covariance,
                             np.dot(observation_matrix.T, np.linalg.inv(innovation_covariance)))

        state = predicted_state + np.dot(kalman_gain, innovation)
        covariance = np.dot(np.eye(n_state_vars) - np.dot(kalman_gain, observation_matrix), predicted_covariance)

        # 保存滤波后的状态
        filtered_states[t] = state

    return filtered_states