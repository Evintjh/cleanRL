import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch.nn import functional as F
import math


class CNNBackbone(nn.Module):
    def __init__(self, nframes, envs):
        super(CNNBackbone, self).__init__()
        self.nframes = nframes
        self.act_fea_cv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=6, padding_mode='circular'
        )
        # Output sequence length: floor((input sequence length + 2 * padding - kernel_size) / stride) + 1
        self.cnn_output_shape = math.floor((360 + 2 * 6 - 5) / 2) + 1
        self.act_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=6, padding_mode='circular'
        )
        self.cnn_output_shape = math.floor((self.cnn_output_shape + 2 * 6 - 3) / 2) + 1
        self.act_fc1 = nn.Linear(self.cnn_output_shape * 32, 256)
        # self.act_fc1 = nn.Linear((int(self.obs_size / 4) + 1 + 7) * 32, 256)
        self.act_fc2 = nn.Linear(256 + 2 * nframes, 128)  # default2, +4 more for WP, +2 for last action
        torch.nn.init.xavier_uniform_(self.act_fc1.weight)
        torch.nn.init.xavier_uniform_(self.act_fc2.weight)
        print('init CNNBackbone')

    def forward(self, lidar_stack, local_goal_stack):
        feat = F.relu(self.act_fea_cv1(lidar_stack))
        feat = F.relu(self.act_fea_cv2(feat))
        feat = feat.flatten(start_dim=1)
        feat = F.relu(self.act_fc1(feat))  # (-1, 256)
        feat = torch.cat((feat, local_goal_stack.flatten(start_dim=1)), dim=-1)  # (-1, 296)
        feat = F.relu(self.act_fc2(feat))
        return feat


class Baseline_Agent(nn.Module):

    def __init__(self, envs, ):
        super(Baseline_Agent, self).__init__()
        self.mu_size = envs._task.dr_size
        self.nframes = envs._task.n_stack
        self.nframes_ori = envs._task.n_stack
        self.backbone_critic = CNNBackbone(self.nframes_ori, envs)
        self.backbone_actor = CNNBackbone(self.nframes_ori, envs)  # bypass obs taking only current frame
        self.crit_fc_value = nn.Linear(128, 1)
        self.act_fc_actor = nn.Linear(128, 2)
        torch.nn.init.xavier_uniform_(self.crit_fc_value.weight)
        torch.nn.init.xavier_uniform_(self.act_fc_actor.weight)
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def get_value(self, obs):
        history_lidar, history_velo, history_local_goal, history_action, current_lidar, curr_velo, curr_local_goal, lidar_stack, local_goal_stack = self.parse_obs(
            obs)

        value = self.backbone_critic(lidar_stack, local_goal_stack)
        value = self.crit_fc_value(value)
        return value

    def get_action_and_value(self, obs, action=None):

        history_lidar, history_velo, history_local_goal, history_action, current_lidar, curr_velo, curr_local_goal, lidar_stack, local_goal_stack = self.parse_obs(
            obs)
        value = self.backbone_critic(lidar_stack, local_goal_stack)
        value = self.crit_fc_value(value)

        feat = self.backbone_actor(lidar_stack, local_goal_stack)
        action_mean = self.act_fc_actor(feat)

        # print("logstd shape: ",self.actor_logstd.shape,"   action mean shape:",action_mean.shape)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:
            action = action.reshape((-1, 2))

        # print(f'action : {action}')
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value, 0

    def get_eval(self, obs):  # TODO clean up
        history_lidar, history_velo, history_local_goal, history_action, current_lidar, curr_velo, curr_local_goal, lidar_stack, local_goal_stack = self.parse_obs(
            obs)

        feat = self.backbone_actor(lidar_stack, local_goal_stack)
        action_mean = self.act_fc_actor(feat)
        return action_mean

    def parse_obs(self, obs):

        mu_size = self.mu_size
        # obs = obs['obs']
        mu = obs[:, -mu_size:]
        obs = obs[:, :-mu_size]
        obs_history = obs[:, :(self.nframes - 1) * 366]
        obs_history_reshaped = obs_history.view(-1, self.nframes - 1, 366)

        # obs_current: [B, 364]
        # obs_current_reshaped: [B, 1, 364]
        obs_current = obs[:, (self.nframes - 1) * 366:]
        obs_current_reshaped = obs_current.view(-1, 1, 364)

        history_lidar, history_velo, history_local_goal, history_action = obs_history_reshaped[:, :,
                                                                          4:364], obs_history_reshaped[:, :,
                                                                                  0:2], obs_history_reshaped[:, :,
                                                                                        2:4], obs_history_reshaped[:, :,
                                                                                              -2:]
        current_lidar, curr_velo, curr_local_goal = obs_current_reshaped[:, :, 4: 364], obs_current_reshaped[:, :,
                                                                                        0:2], obs_current_reshaped[:, :,
                                                                                              2:4]

        if self.nframes_ori == 1:
            lidar_stack = current_lidar
            local_goal_stack = curr_local_goal
        else:
            local_goal_stack = torch.cat((history_local_goal, curr_local_goal), dim=1)
            lidar_stack = torch.cat((history_lidar, current_lidar), dim=1)

        return history_lidar, history_velo, history_local_goal, history_action, current_lidar, curr_velo, curr_local_goal, lidar_stack, local_goal_stack
