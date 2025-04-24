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

from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
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
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            reset_to_state=None,
            domain_shift=None,
            save_name=None,
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
        self.cache_actions = []
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
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
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner shift={self.domain_shift} {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                if first_obs_img is None:
                    first_obs_img = env.call("get_first_obs_frame")[0]
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
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
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
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


def kalman_filter(observations, initial_state, initial_covariance, transition_matrix, observation_matrix, process_covariance, observation_covariance):
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