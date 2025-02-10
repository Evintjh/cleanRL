import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from torch.nn import functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNN1D_BASE(nn.Module):
    def __init__(self, nframes, obs_size, lidar_dim, hidden_dim=32):
        super(CNN1D_BASE, self).__init__()

        self.nframes = nframes
        self.lidar_dim = lidar_dim
        self.conv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='circular'
        )  # 64
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, padding_mode='circular'
        )  # 32

        input_tensor = torch.randn(1, nframes, lidar_dim)
        cnn_feat_dim = np.prod(self.conv2(self.conv1(input_tensor)).shape)

        self.act_fc1 = nn.Linear(cnn_feat_dim, hidden_dim)
        self.act_fc2 = nn.Linear(hidden_dim + hidden_dim, hidden_dim)  # default2, +4 more for WP, +2 for last action
        self.feat_fc1 = nn.Linear(obs_size * nframes, hidden_dim)
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


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        nframe = args.task.env.n_stack
        lidar_dim = args.task.env.lidar_dim
        obs_size = envs.observation_space.shape[1] - lidar_dim

        hidden_dim = args.train.params.config.hidden_dim
        actor_backbone = CNN1D_BASE(nframe, obs_size, lidar_dim, hidden_dim=hidden_dim)
        critic_backbone = CNN1D_BASE(nframe, obs_size, lidar_dim, hidden_dim=hidden_dim)

        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            critic_backbone,
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            actor_backbone,
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, envs.action_space.shape[1]), std=0.01),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.action_space.shape[1]))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_eval(self, x):  # TODO clean up
        action_mean = self.actor_mean(x)
        return action_mean


class AgentMLP(nn.Module):
    def __init__(self, envs, args):
        super().__init__()

        nframe = args.task.env.n_stack
        hidden_dim = args.train.params.config.hidden_dim

        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            layer_init(nn.Linear(envs.observation_space.shape[1] * nframe, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            layer_init(nn.Linear(envs.observation_space.shape[1] * nframe, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, envs.action_space.shape[1]), std=0.01),
        )
        # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        self.actor_logstd = nn.Parameter(torch.zeros(1, envs.action_space.shape[1]))

    def get_value(self, x):
        x = x.flatten(start_dim=1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.flatten(start_dim=1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    def get_eval(self, x):  # TODO clean up
        x = x.flatten(start_dim=1)
        action_mean = self.actor_mean(x)
        return action_mean


class CNNModel(nn.Module):
    def __init__(self, num_lidar_features, num_non_lidar_features, num_actions, nframes=1):
        super(CNNModel, self).__init__()
        self.act_fea_cv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=6, padding_mode='circular'
        )
        self.act_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
        )

        # conv_output_size = (num_lidar_features - 5 + 2*6) // 2 + 1  # Output size after self.act_fea_cv1
        # conv_output_size = (conv_output_size - 3 + 2*1) // 2 + 1  # Output size after self.act_fea_cv2
        # conv_output_size *= 32  # Multiply by the number of output channels
        with torch.no_grad():
            sample_input = torch.randn(1, nframes, num_lidar_features)
            sample_output = self.act_fea_cv1(sample_input)
            sample_output = self.act_fea_cv2(sample_output)
            conv_output_size = sample_output.view(1, -1).shape[1]

        # Calculate the output size of the CNN
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64 + num_non_lidar_features * nframes, 32)
        self.fc3 = nn.Linear(32, num_actions)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, state):
        # lidar = lidar.unsqueeze(1)  # Add channel dimension

        lidar, non_lidar = state[:, :, :360], state[:, :, 360:]

        feat = F.relu(self.act_fea_cv1(lidar))
        feat = F.relu(self.act_fea_cv2(feat))
        feat = feat.view(feat.shape[0], -1)
        feat = F.relu(self.fc1(feat))
        # feat = torch.cat((feat, non_lidar.view(non_lidar.shape[0], -1)), dim=-1)
        feat = torch.cat((feat, non_lidar.flatten(start_dim=1)), dim=-1)
        feat = F.relu(self.fc2(feat))
        feat = self.fc3(feat)
        return feat
