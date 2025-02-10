import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from cleanRL.network.policy_sac import SoftQNetwork, Actor
from cleanRL.trainer import Trainer


class SACTrainer(Trainer):

    def __init__(self, env, args, batch_path, writer, debug, device):

        super().__init__(env, args, batch_path, writer, debug, device)

        self.arg_full = args

        env = self.env
        device = self.device
        self.num_env = self.env._num_envs
        # sac_args = self.sac_args
        sac_args = args.train.params.config
        self.sac_args = sac_args
        self.n_stack = self.arg_full.task.env.n_stack

        self.agent = Actor(env, self.arg_full).to(device)
        self.qf1 = SoftQNetwork(env, self.arg_full).to(device)
        self.qf2 = SoftQNetwork(env, self.arg_full).to(device)
        self.qf1_target = SoftQNetwork(env, self.arg_full).to(device)
        self.qf2_target = SoftQNetwork(env, self.arg_full).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=sac_args.q_lr)
        self.agent_optimizer = optim.Adam(list(self.agent.parameters()), lr=sac_args.policy_lr)

        # Replay Buffer
        env.observation_space.dtype = np.float32
        action_space = env.action_space
        # self.rb = ReplayBuffer(
        #     int(sac_args.buffer_size),
        #     env.observation_space,
        #     action_space,  # env.action_space,
        #     device,
        #     n_envs=int(sac_args.num_envs),
        #     handle_timeout_termination=False, #sac_args.timeout,
        # )

        self.rb = ReplayMemory(state_dim=env.observation_space.shape[1],
                               action_dim=action_space.shape[1],
                               capacity=int(sac_args.buffer_size),
                               num_env=sac_args.num_envs,
                               n_stack=self.n_stack,
                               batch_size=sac_args.batch_size,
                               device=device)

        # Automatic entropy tuning
        # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        self.target_entropy = -action_space.shape[1] * np.log(self.num_env) * sac_args.ent_factor
        print('target_entropy:', self.target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        if sac_args.fixed_alpha:
            self.alpha = sac_args.alpha
        else:
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=sac_args.q_lr)

    def train(self):
        obs = self.env.reset()
        obs = obs['obs']
        for global_step in range(1, int(self.sac_args.total_timesteps) + 1):
            obs = self.rollout(global_step, obs)
            if global_step > self.sac_args.learning_starts:
                for i in range(self.num_env * self.sac_args.steps_per_env):
                    self.update(global_step, i)

        torch.save(self.agent, self.batch_path + '/model.pth')
        self.env.close()

    def rollout(self, global_step, obs):

        # Collect On-Policy Data
        if global_step < self.sac_args.learning_starts:
            actions = self.env.action_space.sample()
            # actions = np.array([self.env.action_space.sample() for _ in range(self.sac_args.num_envs)])
            actions = torch.Tensor(actions).cuda()
        else:
            actions, _, _ = self.agent.get_action(torch.Tensor(obs).to(self.device))
            actions = actions  # .detach().cpu().numpy()

        next_obs, rewards, dones, infos = self.env.step(actions)

        next_obs = next_obs['obs']
        self.reward_sum += torch.mean(rewards)

        real_next_obs = next_obs.clone()
        done_id = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_id) > 0:
            real_next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
            self.env._task.obs_stack = real_next_obs

        self.rb.add(state=obs.detach().cpu().numpy(), action=actions.detach().cpu().numpy(),
                    reward=rewards.detach().cpu().numpy(), next_state=real_next_obs.detach().cpu().numpy(),
                    done=dones.detach().cpu().numpy())

        obs = next_obs
        return obs

    def update(self, global_step, iteration):

        sac_args = self.sac_args
        writer = self.writer

        # data = self.rb.sample(sac_args.batch_size)
        observations, actions, rewards, next_observations, dones = self.rb.sample()
        # print(observations, actions, rewards, next_observations, dones)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.agent.get_action(next_observations)
            qf1_next_target = self.qf1_target(next_observations, next_state_actions)
            qf2_next_target = self.qf2_target(next_observations, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = (rewards.flatten() + (1 - dones.flatten()) * sac_args.gamma * (
                min_qf_next_target).view(-1)).unsqueeze(1)

        qf1_a_values = self.qf1(observations, actions)  # .view(-1)
        qf2_a_values = self.qf2(observations, actions)  # .view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)  # l1_smooth
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        if self.sac_args.grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), self.sac_args.grad_clip_value)
            torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), self.sac_args.grad_clip_value)
        self.q_optimizer.step()

        if global_step % sac_args.policy_frequency == 0:  # TD 3 Delayed update support for policy
            for _ in range(
                    sac_args.policy_frequency):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = self.agent.get_action(observations)
                qf1_pi = self.qf1(observations, pi)
                qf2_pi = self.qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)  # .view(-1)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.agent_optimizer.zero_grad()
                actor_loss.backward()
                if self.sac_args.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.sac_args.grad_clip_value)
                self.agent_optimizer.step()

                if not self.sac_args.fixed_alpha:
                    with torch.no_grad():
                        _, log_pi, _ = self.agent.get_action(observations)
                    alpha_loss = (-self.log_alpha * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if global_step % sac_args.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(sac_args.tau * param.data + (1 - sac_args.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(sac_args.tau * param.data + (1 - sac_args.tau) * target_param.data)

        # Prints
        if global_step % sac_args.write_interval == 0 and iteration == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/next_q_value", next_q_value.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/rewards", rewards.mean().item(), global_step)
            writer.add_scalar("1returns/step_reward", self.reward_sum / sac_args.write_interval, global_step)
            writer.add_scalar("losses/alpha", self.alpha, global_step)
            if not self.sac_args.fixed_alpha:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

            result_stat = self.env._task.result_stat
            writer.add_scalar("results/timeout", np.mean(result_stat[0, :]), global_step)
            writer.add_scalar("results/reachgoal", np.mean(result_stat[1, :]), global_step)
            writer.add_scalar("results/collision", np.mean(result_stat[2, :]), global_step)
            if len(self.env._task.ep_len_stat) > 0:
                writer.add_scalar("results/ep_len", np.mean(self.env._task.ep_len_stat), global_step)

            fps = int(global_step / (time.time() - self.start_time))
            writer.add_scalar("zothers/FPS", fps, global_step)
            print("Step %d:, Average returns: %.3f, with %d FPS" % (
                global_step, self.reward_sum / sac_args.write_interval, fps))
            self.reward_sum = 0

    def eval(self, env):
        obs = env.reset()
        eval_ep = 0
        ep_reward = []
        while eval_ep < int(self.sac_args.eval_episodes):
            actions = self.agent.get_eval(torch.Tensor(obs).to(self.device))
            actions = actions.detach().cpu().numpy()
            next_obs, rewards, dones, infos = env.step(actions)

            if dones:
                eval_ep += 1
                obs = [infos[0]["terminal_observation"]]
                ep_reward.append(infos[0]["episode"]["r"])
            else:
                obs = next_obs

        return ep_reward


class ReplayMemory():
    """Buffer to store environment transitions."""

    def __init__(self, state_dim, action_dim, capacity, num_env, n_stack, batch_size, device):
        self.capacity = int(capacity)
        self.device = device
        self.num_env = num_env
        self.batch_size = batch_size

        print('capacity', self.capacity)
        print('state_dim', state_dim)
        print('action_dim', action_dim)

        self.states = np.empty((self.capacity, n_stack, int(state_dim)), dtype=np.float32)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, n_stack, int(state_dim)), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

        # Pre-allocate pinned memory for sampling
        self.sample_states = torch.zeros((batch_size, n_stack, int(state_dim)), dtype=torch.float32, pin_memory=True)
        self.sample_actions = torch.zeros((batch_size, int(action_dim)), dtype=torch.float32, pin_memory=True)
        self.sample_rewards = torch.zeros((batch_size, 1), dtype=torch.float32, pin_memory=True)
        self.sample_next_states = torch.zeros((batch_size, n_stack, int(state_dim)), dtype=torch.float32,
                                              pin_memory=True)
        self.sample_dones = torch.zeros((batch_size, 1), dtype=torch.float32, pin_memory=True)

    def add(self, state, action, reward, next_state, done):
        for state_, action_, reward_, next_state_, done_ in zip(state, action, reward, next_state, done):
            np.copyto(self.states[self.idx], state_)
            np.copyto(self.actions[self.idx], action_)
            np.copyto(self.rewards[self.idx], reward_)
            np.copyto(self.next_states[self.idx], next_state_)
            np.copyto(self.dones[self.idx], done_)
            # print(self.idx, state_, action_, reward_, next_state_, done_)

            self.idx = (self.idx + 1) % self.capacity
            self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        # Copy data to pre-allocated tensors
        np.copyto(self.sample_states.numpy(), self.states[idxs])
        np.copyto(self.sample_actions.numpy(), self.actions[idxs])
        np.copyto(self.sample_rewards.numpy(), self.rewards[idxs])
        np.copyto(self.sample_next_states.numpy(), self.next_states[idxs])
        np.copyto(self.sample_dones.numpy(), self.dones[idxs])

        # Move pre-allocated tensors to device
        return (
            self.sample_states.to(self.device, non_blocking=True),
            self.sample_actions.to(self.device, non_blocking=True),
            self.sample_rewards.to(self.device, non_blocking=True),
            self.sample_next_states.to(self.device, non_blocking=True),
            self.sample_dones.to(self.device, non_blocking=True)
        )
