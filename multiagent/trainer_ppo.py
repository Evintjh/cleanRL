import time
import copy
import numpy as np
import torch
import torch.optim as optim
from cleanRL.network.policy_ppo import Agent, AgentMLP, CNNModel
from cleanRL.trainer import Trainer
import torch.nn as nn
import pickle
from collections import deque


class PPOTrainer(Trainer):

    def __init__(self, env, args, batch_path, writer, debug, device):

        super().__init__(env, args, batch_path, writer, debug, device)

        self.arg_full = args
        self.args = args.train.params.config

        env = self.env
        device = self.device
        args = self.args
        self.n_stack = self.arg_full.task.env.n_stack
        self.time_lag = self.arg_full.task.env.time_lag

        self.agent = Agent(env, self.arg_full).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=args.ppo_lr, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs, self.n_stack, env.observation_space.shape[1])).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs, env.action_space.shape[1])).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.returns = None
        self.advantages = None

        self.batch_size = int(args.num_envs * args.num_steps)
        self.minibatch_size = int(self.batch_size // args.num_minibatches)
        self.num_updates = int(args.total_timesteps // self.batch_size)  # * args.num_envs
        self.reward_sum = 0

        # Time lag
        self.action_stack = deque(maxlen=(1 + self.time_lag))
        for _ in range(self.time_lag):
            self.action_stack.append(self.zero_action())

    def zero_action(self, num_env=None):
        if num_env is None:
            num_env = self.args.num_envs
        return torch.zeros((num_env, self.env.action_space.shape[1])).to(self.device)

    def train(self):

        args = self.args
        device = self.device

        next_obs = self.env.reset()
        next_obs = next_obs['obs']
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        global_step = 0
        for update_i in range(1, self.num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update_i - 1.0) / self.num_updates
                lrnow = frac * args.ppo_lr
                self.optimizer.param_groups[0]["lr"] = lrnow
            global_step, next_obs, next_done = self.rollout(global_step, next_obs, next_done)
            self.update(global_step, update_i)

            if update_i % args.save_interval == 0 and not self.debug:
                torch.save(self.agent.state_dict(), self.batch_path + '/model/model_%04d.pth' % update_i)

        torch.save(self.agent.state_dict(), self.batch_path + '/model/model.pth')
        self.env.close()

    def rollout(self, global_step, next_obs, next_done):

        args = self.args
        device = self.device

        for step in range(0, args.num_steps):

            # Use reseted states after reset. Init nstack for stacked states
            done_id = next_done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_id) > 0:
                next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
                self.env._task.obs_stack = next_obs

                for lag_i in range(self.time_lag):
                    self.action_stack[lag_i][done_id] = self.zero_action(len(done_id))

            global_step += 1 * args.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action_, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()

            # Apply time lag
            self.action_stack.append(action_)
            action = self.action_stack[0]

            self.actions[step] = action_
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs['obs']

            # print(step, self.obs[step][:, :, -4].cpu().numpy(), next_obs[:, :, -4].cpu().numpy(), done.cpu().numpy())

            self.reward_sum += torch.mean(reward)
            self.rewards[step] = reward.view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[
                    t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

            self.returns = returns
            self.advantages = advantages

        return global_step, next_obs, next_done

    def update(self, global_step, update_i):

        args = self.args
        writer = self.writer

        # flatten the batch
        b_obs = self.obs.reshape(-1, self.n_stack, self.env.observation_space.shape[1])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1, self.env.action_space.shape[1])
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                self.optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        result_stat = self.env._task.result_stat
        writer.add_scalar("results/timeout", np.mean(result_stat[0, :]), global_step)
        writer.add_scalar("results/reachgoal", np.mean(result_stat[1, :]), global_step)
        writer.add_scalar("results/collision", np.mean(result_stat[2, :]), global_step)
        if len(self.env._task.ep_len_stat) > 0:
            writer.add_scalar("results/ep_len", np.mean(self.env._task.ep_len_stat), global_step)

        fps = int(global_step / (time.time() - self.start_time))
        writer.add_scalar("zothers/FPS", fps, global_step)
        if update_i % args.ppo_print_interval == 0:
            reward_print = self.reward_sum / (args.num_steps * args.ppo_print_interval)
            writer.add_scalar("losses/returns", reward_print, global_step)
            print("Step %d / %d:, Average returns: %.2f, with %d FPS" % (
                update_i, self.num_updates, reward_print, fps))
            self.reward_sum = 0

    def eval(self, env, model_path, global_step=0):

        args = self.args
        device = self.device
        model_weights = torch.load('results/Jackal/' + self.arg_full.model_path, weights_only=True)
        self.agent.load_state_dict(model_weights)

        next_obs = self.env.reset()
        next_obs = next_obs['obs']
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        env_data = {
            env_id: {
                "obs": [],  # Initialize with an empty list (or another default value)
                "action": [],  # Initialize with an empty list (or another default value)
                "next_obs": [],  # Initialize with an empty list (or another default value)
                "reward": [],  # Initialize with an empty list (or another default value)
                "done": []  # Initialize with an empty list (or another default value)
            }
            for env_id in range(args.num_envs)
        }
        full_data = []

        result_stat = self.env._task.reset_stats(num_ep_eval=args.eval_episodes)
        step = 0

        while 0 in np.sum(result_stat, axis=0) and len(full_data) < 1000:
            # print(step)
            step += 1

            done_id = next_done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_id) > 0:
                next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
                self.env._task.obs_stack = next_obs
                # print(result_stat)
                # print()

                if self.arg_full.save_data:
                    for env_id in done_id.detach().cpu().numpy():
                        episode_data = copy.deepcopy(env_data[env_id])  # Create a deep copy
                        full_data.append(episode_data)  # Add the episode data to the list
                        env_data[env_id] = {
                            "obs": [],  # Initialize with an empty list (or another default value)
                            "action": [],  # Initialize with an empty list (or another default value)
                            "next_obs": [],  # Initialize with an empty list (or another default value)
                            "reward": [],  # Initialize with an empty list (or another default value)
                            "done": []  # Initialize with an empty list (or another default value)
                        }

            with torch.no_grad():
                action = self.agent.get_eval(next_obs)

            obs = next_obs

            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs['obs']

            if self.arg_full.save_data:
                for env_id in range(args.num_envs):
                    env_data[env_id]["obs"].append(obs[env_id].detach().cpu().numpy())
                    env_data[env_id]["action"].append(action[env_id].detach().cpu().numpy())
                    env_data[env_id]["next_obs"].append(next_obs[env_id].detach().cpu().numpy())
                    env_data[env_id]["reward"].append(reward[env_id].detach().cpu().numpy())
                    env_data[env_id]["done"].append(done[env_id].detach().cpu().numpy())

            self.reward_sum += torch.mean(reward)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        # Save all_episodes to disk
        if self.arg_full.save_data:
            with open('full_data2.pkl', 'wb') as f:
                pickle.dump(full_data, f)

        print('total ep', self.env._task.result_stat.shape[1])
        print("timeout", np.mean(result_stat[0, :]))
        print("reachgoal", np.mean(result_stat[1, :]))
        print("collision", np.mean(result_stat[2, :]))
        if len(self.env._task.ep_len_stat) > 0:
            print("ep_len", np.mean(self.env._task.ep_len_stat))

    def eval_offline(self, env, model_path, global_step=0):

        args = self.args
        device = self.device
        model_weights = torch.load('cnn_model.pth', weights_only=True)

        self.agent = CNNModel(360, 2, 2, 3).to(device)

        self.agent.load_state_dict(model_weights)

        next_obs = self.env.reset()
        next_obs = next_obs['obs']
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        result_stat = self.env._task.reset_stats(num_ep_eval=args.eval_episodes)
        step = 0

        while 0 in np.sum(result_stat, axis=0):
            print(step)
            step += 1

            done_id = next_done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_id) > 0:
                next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
                self.env._task.obs_stack = next_obs

            with torch.no_grad():
                action = self.agent(next_obs)

            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs['obs']

            self.reward_sum += torch.mean(reward)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        print('total ep', self.env._task.result_stat.shape[1])
        print("timeout", np.mean(result_stat[0, :]))
        print("reachgoal", np.mean(result_stat[1, :]))
        print("collision", np.mean(result_stat[2, :]))
        if len(self.env._task.ep_len_stat) > 0:
            print("ep_len", np.mean(self.env._task.ep_len_stat))

    def eval_diffusion(self, env, model_path, global_step=0):
        from saved_diffusion.diffusion_model import load_policy

        args = self.args
        device = self.device

        self.agent = load_policy('saved_diffusion/diffuser_policy_3.pth', device=device)

        next_obs = self.env.reset()
        next_obs = next_obs['obs']
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        result_stat = self.env._task.reset_stats(num_ep_eval=args.eval_episodes)
        step = 0
        batch = {}

        self.agent.eval()

        while 0 in np.sum(result_stat, axis=0):
            print(step)
            step += 1

            done_id = next_done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_id) > 0:
                next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
                self.env._task.obs_stack = next_obs

            with torch.no_grad():
                # action = self.agent(next_obs)
                batch['obs'] = next_obs.reshape(next_obs.shape[0], -1).unsqueeze(1)
                action = self.agent.predict_action(batch)
                action = action['action'][:, 0]

            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs['obs']

            self.reward_sum += torch.mean(reward)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        print('total ep', self.env._task.result_stat.shape[1])
        print("timeout", np.mean(result_stat[0, :]))
        print("reachgoal", np.mean(result_stat[1, :]))
        print("collision", np.mean(result_stat[2, :]))
        if len(self.env._task.ep_len_stat) > 0:
            print("ep_len", np.mean(self.env._task.ep_len_stat))

    def eval_diffusion_cnn(self, env, model_path, global_step=0):
        from saved_diffusion_cnn.diffusion_policy_cnn_model import load_policy

        args = self.args
        device = self.device

        self.agent = load_policy('saved_diffusion_cnn/diffusion_policy_cnn.pth', device=device)

        next_obs = self.env.reset()
        next_obs = next_obs['obs']
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        result_stat = self.env._task.reset_stats(num_ep_eval=args.eval_episodes)
        step = 0
        batch = {}

        self.agent.eval()

        while 0 in np.sum(result_stat, axis=0):
            print(step)
            step += 1

            done_id = next_done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_id) > 0:
                next_obs[done_id] = self.env._task.extras['first_obs'].unsqueeze(1).repeat(1, self.n_stack, 1)
                self.env._task.obs_stack = next_obs

            with torch.no_grad():
                # print(next_obs.shape)

                batch['lidar_data'] = next_obs[:, :, :360].unsqueeze(1)
                batch['non_lidar_data'] = next_obs[:, :, 360:].unsqueeze(1)

                # batch['obs'] = next_obs.reshape(next_obs.shape[0], -1).unsqueeze(1)
                action = self.agent.predict_action(batch['lidar_data'], batch['non_lidar_data'])
                action = action['action'][:, 0]

            next_obs, reward, done, infos = self.env.step(action)
            next_obs = next_obs['obs']

            self.reward_sum += torch.mean(reward)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        print('total ep', self.env._task.result_stat.shape[1])
        print("timeout", np.mean(result_stat[0, :]))
        print("reachgoal", np.mean(result_stat[1, :]))
        print("collision", np.mean(result_stat[2, :]))
        if len(self.env._task.ep_len_stat) > 0:
            print("ep_len", np.mean(self.env._task.ep_len_stat))
