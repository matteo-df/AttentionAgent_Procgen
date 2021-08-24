import os

import gym3
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import matplotlib as mpl

from src.common.utils import patch_center_position, divide_in_patches


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
        img = Image.fromarray(self.env.get_info()[0]['rgb'], 'RGB').resize(
            (self.hparams.resized_size, self.hparams.resized_size))
        img = ToTensor()(img)

        patches = divide_in_patches(img, self.hparams.patch_size, self.hparams.stride_patches)

        top_k_idx = self.get_top_k_patches_idx(patches)

        feature = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                        stride=self.hparams.stride_patches)
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
        # if self.hparams.feature_retrieval == 'patch_center_position':
        feature = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                        stride=self.hparams.stride_patches)
        return feature

    def evaluate(self):
        done = False
        total_reward = 0.0
        while not done:
            total_reward -= self.hparams.surviving_penalty
            action = self()
            self.env.act(action)
            reward, _, done = self.env.observe()
            total_reward += reward.item()
        return np.round(total_reward, 3)

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
            rew_5x = 0
            for _ in range(5):
                rew_5x += self.evaluate()
            tot_rewards.append(rew_5x/5)
        return np.asarray(tot_rewards)

    def save_model(self):
        model_path = self.hparams.log.log_model + self.hparams.log.exp_name + '.pth'
        torch.save(self.state_dict(), model_path)

    def load_model(self, pretrained_model):
        assert os.path.exists(pretrained_model), 'Pretrained model not found!'
        self.load_state_dict(torch.load(pretrained_model))

    def test(self):
        total_reward = 0
        while total_reward == 0:
            done = False
            gif = []
            while not done:
                frame = Image.fromarray(self.env.get_info()[0]['rgb'], 'RGB')

                resized_frame = frame.resize((self.hparams.resized_size, self.hparams.resized_size))
                resized_frame = ToTensor()(resized_frame)
                patches = divide_in_patches(resized_frame, self.hparams.patch_size, self.hparams.stride_patches)
                top_k_idx = self.get_top_k_patches_idx(patches)
                patch_centers = patch_center_position(patches, original_img_size=self.hparams.resized_size,
                                                      stride=self.hparams.stride_patches)[top_k_idx]
                frame = frame.resize((5 * self.hparams.resized_size, 5 * self.hparams.resized_size))
                overplot = ImageDraw.Draw(frame)

                cmap = mpl.cm.autumn
                for j, center in enumerate(patch_centers):
                    xc, yc = 5 * center
                    x0, y0 = xc - 5 * self.hparams.stride_patches, yc - 5 * self.hparams.stride_patches
                    x1, y1 = xc + 5 * self.hparams.stride_patches, yc + 5 * self.hparams.stride_patches
                    color = mpl.colors.rgb2hex(cmap(j / 10.))
                    overplot.rectangle([x0, y0, x1, y1], outline=color, width=2)

                reward_text = 'REWARD: ' + str(total_reward)
                font_path = self.hparams.log.home + '/src/common/Font/GidoleFont/Gidole-Regular.ttf'
                font = ImageFont.truetype(font_path, 18)
                overplot.text((10, 452), reward_text, font=font)

                action = self()
                self.env.act(action)
                rew, _, done = self.env.observe()
                total_reward += rew.item()

                gif.append(frame)
        gif[0].save(self.hparams.log.log_gif + '/temp_result.gif', save_all=True, optimize=False, append_images=gif[1:],
                    loop=0)

    def test_on_screen(self):
        self.env = gym3.ViewerWrapper(self.env, info_key="rgb")
        while 1:
            action = self()
            self.env.act(action)
