import copy
import os

import gym3
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import matplotlib.cm
import matplotlib.colors

from tqdm import tqdm

from src.common.utils import patch_center_position, divide_in_patches, random_color_in_patch


class AttentionAgent(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Instantiating the Procgen Environment
        env = hydra.utils.instantiate(self.hparams.game)
        self.env = gym3.Wrapper(env)

        # Instantiating the Self Attention module
        self.hparams.self_attention.data_dim = 3 * self.hparams.patch_size ** 2
        self.attention = hydra.utils.instantiate(self.hparams.self_attention)

        # Instantiating the LSTM module
        self.lstm = hydra.utils.instantiate(self.hparams.lstm)

    def forward(self):
        act = self.get_action()
        return act

    def get_action(self, env=None):
        if env is None:
            env = self.env
        img = Image.fromarray(env.get_info()[0]['rgb'], 'RGB').resize(
            (self.hparams.resized_size, self.hparams.resized_size))
        img = ToTensor()(img)

        patches = divide_in_patches(img, self.hparams.patch_size, self.hparams.stride_patches)

        top_k_idx = self.get_top_k_patches_idx(patches)

        feature = self.get_patch_features(patches)
        feature = feature[top_k_idx]
        feature = feature.flatten(0, 1)[None, ...]

        act = self.lstm(feature)
        act = torch.argmax(act).numpy().reshape(1, )

        return act

    def get_top_k_patches_idx(self, patches):
        flattened_patches = patches.reshape(1, -1, 3 * self.hparams.patch_size ** 2)
        patch_importance_matrix = self.attention(flattened_patches)
        patch_importance = patch_importance_matrix.sum(dim=0)
        # extract top k important patches
        idx = torch.argsort(patch_importance, descending=True)
        top_k_idx = idx[:self.hparams.top_k]
        return top_k_idx

    def get_patch_features(self, patches):
        if self.hparams.feature_retrieval == 'position':
            feature = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                            stride=self.hparams.stride_patches)

        elif self.hparams.feature_retrieval == 'position+color':
            pos = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                        stride=self.hparams.stride_patches)
            rand_col = random_color_in_patch(patches)
            feature = torch.cat((pos, rand_col[:, None]), dim=1)

        else:
            raise ValueError

        return feature

    def evaluate(self):
        done = False
        total_reward = 0.0
        total_penalty = 0.0
        steps = 0
        while not done and steps < 500:
            total_penalty += self.hparams.surviving_penalty
            action = self()
            self.env.act(action)
            reward, _, done = self.env.observe()
            total_reward += reward.item()
            steps += 1
            if done:
                if reward.item() == 10:
                    total_reward = 20
                else:
                    total_reward += -3
            if steps == 500:
                total_reward += -1
        return np.round(total_reward - total_penalty, 3)

    def evaluate_no_penalty(self):
        done = False
        total_reward = 0.0
        while not done:
            action = self.get_action(self.env)
            self.env.act(action)
            reward, _, done = self.env.observe()
            total_reward += reward.item()
        return total_reward

    def get_params(self):
        params = []
        weight_dict = self.state_dict()
        for k in sorted(weight_dict.keys()):
            params.append(weight_dict[k].numpy().copy().ravel())
        return np.concatenate(params)

    def set_params(self, parameters):
        offset = 0
        weights_to_set = {}
        weight_dict = self.state_dict()
        for k in sorted(weight_dict.keys()):
            weight = weight_dict[k].numpy()
            weight_size = weight.size
            weights_to_set[k] = torch.from_numpy(
                parameters[offset:(offset + weight_size)].reshape(weight.shape))
            offset += weight_size
        self.load_state_dict(state_dict=weights_to_set)

    def evaluate_population(self, param_population):
        assert len(param_population.shape) == 2, 'A list of parameters is required'
        list_lenght = param_population.shape[0]
        tot_rewards = []
        for i in range(list_lenght):
            self.set_params(param_population[i])
            rew_5x = []
            for _ in range(5):
                rew_5x.append(self.evaluate())
            rew_5x.sort(reverse=True)
            tot_rewards.append(rew_5x[0])
        return np.asarray(tot_rewards)

    def save_model(self):
        model_path = self.hparams.log.log_model + self.hparams.log.agent_name + '.pth'
        torch.save(self.state_dict(), model_path)

    def load_model(self, pretrained_model):
        assert os.path.exists(pretrained_model), 'Pretrained model not found!'
        self.load_state_dict(torch.load(pretrained_model))

    def test(self, n_episodes=100):
        if n_episodes != 0:
            rews = np.zeros(n_episodes)
            for i in tqdm(range(n_episodes), desc='Playing some games...'):
                rews[i] = self.evaluate_no_penalty()
            at_least_1 = sum(rews > 1.99)
            at_least_2 = sum(rews > 3.99)
            at_least_3 = sum(rews > 5.99)
            at_least_4 = sum(rews > 7.99)
            win = sum(rews > 12.99)

            print('At least 1 enemy defeated:', at_least_1, '/', n_episodes, '->',   np.round(100 * at_least_1 / n_episodes, 1), '%')
            print('At least 2 enemies defeated:', at_least_2, '/', n_episodes, '->', np.round(100 * at_least_2 / n_episodes, 1), '%')
            print('At least 3 enemies defeated:', at_least_3, '/', n_episodes, '->', np.round(100 * at_least_3 / n_episodes, 1), '%')
            print('At least 4 enemies defeated or win:', at_least_4, '/', n_episodes, '->', np.round(100 * at_least_4 / n_episodes, 1),
                  '%')
            print('Wins:', win, '/', n_episodes, '->', np.round(100 * win / n_episodes, 1), '%')

    def save_gif_with_attention_patches(self):
        best_reward = -1.0
        gif = []
        for _ in tqdm(range(10), desc='Playing some games...'):
            seed = np.random.randint(0, 1000000)

            simple_cfg = copy.deepcopy(self.hparams.game)
            simple_cfg.num_levels = 1
            simple_cfg.start_level = seed
            original_cfg = copy.deepcopy(self.hparams.game)
            original_cfg.num_levels = 1
            original_cfg.start_level = seed
            original_cfg.use_backgrounds = True
            original_cfg.use_monochrome_assets = False
            original_cfg.restrict_themes = False
            simple_env = hydra.utils.instantiate(simple_cfg)
            simple_env = gym3.Wrapper(simple_env)
            original_env = hydra.utils.instantiate(original_cfg)
            original_env = gym3.Wrapper(original_env)

            done = False
            tmp_gif = []
            tmp_reward = 0.0
            while not done:
                frame = Image.fromarray(simple_env.get_info()[0]['rgb'], 'RGB')
                original_frame = Image.fromarray(original_env.get_info()[0]['rgb'], 'RGB')

                resized_frame = frame.resize((self.hparams.resized_size, self.hparams.resized_size))
                resized_frame = ToTensor()(resized_frame)
                patches = divide_in_patches(resized_frame, self.hparams.patch_size, self.hparams.stride_patches)
                top_k_idx = self.get_top_k_patches_idx(patches)
                patch_centers = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                                      stride=self.hparams.stride_patches)[top_k_idx]
                original_frame = original_frame.resize((5 * self.hparams.resized_size, 5 * self.hparams.resized_size))
                overplot = ImageDraw.Draw(original_frame)

                cmap = matplotlib.cm.autumn
                for j, center in enumerate(patch_centers):
                    xc, yc = 5 * center
                    x0, y0 = xc - 5 * self.hparams.stride_patches, yc - 5 * self.hparams.stride_patches
                    x1, y1 = xc + 5 * self.hparams.stride_patches, yc + 5 * self.hparams.stride_patches
                    color = matplotlib.colors.rgb2hex(cmap(j / 10.))
                    overplot.rectangle([x0, y0, x1, y1], outline=color, width=2)

                reward_text = 'REWARD: ' + str(tmp_reward)
                font_path = self.hparams.log.home + '/src/common/Font/GidoleFont/Gidole-Regular.ttf'
                font = ImageFont.truetype(font_path, 18)
                overplot.text((10, 452), reward_text, font=font)

                action = self.get_action(simple_env)
                simple_env.act(action)
                original_env.act(action)
                rew, _, done = simple_env.observe()
                tmp_reward += rew.item()

                tmp_gif.append(original_frame)
            if tmp_reward > best_reward:
                best_reward = tmp_reward
                gif = tmp_gif

        gif[0].save(self.hparams.log.log_gif + self.hparams.log.agent_name + '_rew' + str(int(best_reward)) + '.gif', save_all=True, optimize=False,
                    append_images=gif[1:],
                    loop=0)

    def play_on_screen(self, n_episodes=3):
        for _ in range(n_episodes):
            seed = np.random.randint(0, 1000000)

            simple_cfg = copy.deepcopy(self.hparams.game)
            simple_cfg.num_levels = 1
            simple_cfg.start_level = seed
            original_cfg = copy.deepcopy(self.hparams.game)
            original_cfg.num_levels = 1
            original_cfg.start_level = seed
            original_cfg.use_backgrounds = True
            original_cfg.use_monochrome_assets = False
            original_cfg.restrict_themes = False

            simple_env = hydra.utils.instantiate(simple_cfg)
            simple_env = gym3.Wrapper(simple_env)

            original_env = hydra.utils.instantiate(original_cfg)
            original_env = gym3.ViewerWrapper(original_env, info_key="rgb")
            done = False
            while not done:
                action = self.get_action(env=simple_env)
                simple_env.act(action)
                _, _, done = simple_env.observe()
                original_env.act(action)
            original_env._renderer._glfw.destroy_window(original_env._renderer._window)

    def same_frame_two_views(self):
        seed = 42

        simple_cfg = copy.deepcopy(self.hparams.game)
        simple_cfg.num_levels = 1
        simple_cfg.start_level = seed
        original_cfg = copy.deepcopy(self.hparams.game)
        original_cfg.num_levels = 1
        original_cfg.start_level = seed
        original_cfg.use_backgrounds = True
        original_cfg.use_monochrome_assets = False
        original_cfg.restrict_themes = False

        simple_env = hydra.utils.instantiate(simple_cfg)
        simple_env = gym3.Wrapper(simple_env)

        original_env = hydra.utils.instantiate(original_cfg)
        original_env = gym3.Wrapper(original_env)

        for _ in range(20):
            action = self.get_action(env=simple_env)
            simple_env.act(action)
            original_env.act(action)

        frame_original = Image.fromarray(original_env.get_info()[0]['rgb'], 'RGB')
        frame_simplified = Image.fromarray(simple_env.get_info()[0]['rgb'], 'RGB')
        frame_original.save(self.hparams.log.log_dir + '/frame_original.png')
        frame_simplified.save(self.hparams.log.log_dir + '/frame_simplified.png')
