import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class CNN1D_BASE(nn.Module):
    def __init__(self, nframes, obs_size, lidar_dim, out_dim=32):
        super(CNN1D_BASE, self).__init__()

        self.nframes = nframes
        self.lidar_dim = lidar_dim
        self.conv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='circular'
        )
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='circular'
        )

        input_tensor = torch.randn(1, nframes, lidar_dim)
        cnn_feat_dim = np.prod(self.conv2(self.conv1(input_tensor)).shape)

        self.act_fc1 = nn.Linear(cnn_feat_dim, 32)
        self.act_fc2 = nn.Linear(32 + 32, out_dim)  # default2, +4 more for WP, +2 for last action
        self.feat_fc1 = nn.Linear(obs_size * nframes, 32)
        torch.nn.init.xavier_uniform_(self.act_fc1.weight)
        torch.nn.init.xavier_uniform_(self.act_fc2.weight)
        torch.nn.init.xavier_uniform_(self.feat_fc1.weight)
        print('init CNN1D_BASE')

    def forward(self, state):
        x, feat = state[:, :, :self.lidar_dim], state[:, :, self.lidar_dim:]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.act_fc1(x))

        feat = feat.flatten(start_dim=1)
        feat = F.relu(self.feat_fc1(feat))

        x = torch.cat((x, feat), dim=-1)
        x = F.relu(self.act_fc2(x))
        return x


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        nframe = args.task.env.n_stack
        lidar_dim = args.task.env.lidar_dim
        obs_size = env.observation_space.shape[1] - lidar_dim
        self.backbone = CNN1D_BASE(nframe, obs_size, lidar_dim)

        # self.fc1 = nn.Linear(np.prod(env.observation_space.shape) + np.prod(env.action_space.shape), 256)
        self.fc1 = nn.Linear(32 + env.action_space.shape[1], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = self.backbone(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        nframe = args.task.env.n_stack
        lidar_dim = args.task.env.lidar_dim
        obs_size = env.observation_space.shape[1] - lidar_dim
        self.backbone = CNN1D_BASE(nframe, obs_size, lidar_dim)

        # self.fc1 = nn.Linear(env.observation_space.shape[1], 256)  #
        self.fc1 = nn.Linear(32, 256)  #
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, env.action_space.shape[1])
        self.fc_logstd = nn.Linear(256, env.action_space.shape[1])
        # action rescaling
        low = np.repeat(env.action_space.low[0], 1)
        high = np.repeat(env.action_space.high[0], 1)
        self.register_buffer(
            "action_scale",
            torch.tensor((high - low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((high + low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_eval(self, x):  # TODO clean up
        mean, log_std = self(x)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return mean
