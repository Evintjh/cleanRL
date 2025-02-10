import time
import numpy as np
import torch
import torch.optim as optim
from cleanRL.network.Baseline import Baseline_Agent
from cleanRL.network.UP_OSI_v2 import UP_OSI_v2_Agent
from cleanRL.trainer import Trainer
import torch.nn as nn
from collections import deque
import copy


class PPOTrainer(Trainer):

    def __init__(self, env, args, batch_path, writer):

        super().__init__(env, args, batch_path, writer)
        Agent_list = {"Baseline": Baseline_Agent, "UP_OSI_v2": UP_OSI_v2_Agent}

        env = self.env
        device = self.device
        args = self.args
        ppo_args = args.train.params.config
        self.ppo_args = ppo_args
        network = args.train.params.network

        self.agent = Agent_list[network](env).to(device)
        torch.save(self.agent, self.batch_path + f'/model/model_test' + '.pth')
        self.optimizer = optim.Adam(self.agent.parameters(), lr=ppo_args.learning_rate, eps=1e-5)
        self.MSE_loss = nn.MSELoss()

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((ppo_args.horizon_length, args.num_envs, env.observation_space.shape[1])).to(device)
        print("obs:", self.obs.shape)
        self.actions = torch.zeros((ppo_args.horizon_length, args.num_envs, env.action_space.shape[1])).to(device)
        print("actions:", self.actions.shape)
        self.logprobs = torch.zeros((ppo_args.horizon_length, args.num_envs)).to(device)
        self.rewards = torch.zeros((ppo_args.horizon_length, args.num_envs)).to(device)
        self.dones = torch.zeros((ppo_args.horizon_length, args.num_envs)).to(device)
        self.values = torch.zeros((ppo_args.horizon_length, args.num_envs)).to(device)
        self.returns = None
        self.advantages = None
        self.success_array = np.empty((args.num_envs,), dtype=object)
        self.timeout_array = np.empty((args.num_envs,), dtype=object)
        self.collision_array = np.empty((args.num_envs,), dtype=object)
        for i in range(args.num_envs):
            self.success_array[i] = deque(maxlen=20)
            self.timeout_array[i] = deque(maxlen=20)
            self.collision_array[i] = deque(maxlen=20)

        self.batch_size = int(ppo_args.horizon_length * args.num_envs)
        self.minibatch_size = int(self.batch_size // ppo_args.num_minibatches)
        self.num_updates = ppo_args.max_epochs
        self.reward_sum = 0
        self.model_save_interval = ppo_args.save_frequency
        self.eval_interval = ppo_args.eval_frequency
        self.total_eval_ep = 0
        self.ep_rewards_buf = []
        self.ep_length_buf = []
        if ppo_args.debug_mode:
            self.model_save_interval = 1
            self.eval_interval = 1
            self.num_updates = 1
            ppo_args.eval_episodes = 1
            ppo_args.horizon_length = 100

        self.ep_rewards = np.zeros((args.num_envs,), dtype=object)
        for i in range(args.num_envs):
            self.ep_rewards[i] = []

    def train(self):

        args = self.args
        ppo_args = self.ppo_args
        device = self.device

        next_obs = self.env.reset()
        next_obs = torch.Tensor(next_obs['obs']).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        global_step = 0
        for update_i in range(1, self.num_updates + 1):

            # Annealing the rate if instructed to do so.
            if ppo_args.lr_schedule:
                frac = 1.0 - (update_i - 1.0) / self.num_updates
                lrnow = frac * args.ppo_lr
                self.optimizer.param_groups[0]["lr"] = lrnow

            start_time = time.time()
            global_step, next_obs, next_done, tasks_results_buf = self.rollout(global_step, next_obs, next_done)
            rollout_duration = time.time() - start_time
            # print(f"Rollout function took {rollout_duration} seconds.")

            start_time = time.time()
            self.update(global_step, update_i, tasks_results_buf)
            update_duration = time.time() - start_time
            # print(f"Update function took {update_duration} seconds.")
            if update_i % self.model_save_interval == 0:
                torch.save(self.agent.state_dict(), self.batch_path + f'/model/model_{global_step}' + '.pth')
                self.model_save_interval += ppo_args.save_frequency

            if update_i % self.eval_interval == 0:
                results = self.eval(self.env, agent=self.agent, global_step=global_step)
                print('results', results)
                self.eval_interval += ppo_args.eval_frequency

        torch.save(self.agent.state_dict(), self.batch_path + f'/model/model_{global_step}' + '.pth')
        self.env.close()

    def rollout(self, global_step, next_obs, next_done):

        args = self.args
        ppo_args = self.ppo_args
        device = self.device

        for step in range(0, ppo_args.horizon_length):
            global_step += 1 * args.num_envs

            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, mu_hat = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.

            start_time = time.time()
            next_obs, reward, done, infos = self.env.step(action)
            rollout_duration = time.time() - start_time
            # print(f"Rollout function took {rollout_duration} seconds.")

            self.reward_sum += torch.mean(reward)
            self.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs['obs']).to(device), torch.Tensor(done).to(device)

            done_indices = torch.nonzero(done == 1).squeeze()

            for i in range(args.num_envs):
                self.ep_rewards[i].append(copy.deepcopy(reward[i]))
                if done[i]:
                    self.ep_rewards_buf.append(sum(self.ep_rewards[i]))
                    self.ep_length_buf.append(len(self.ep_rewards[i]))
                    self.ep_rewards[i] = []

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(ppo_args.horizon_length)):
                if t == ppo_args.horizon_length - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + ppo_args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[
                    t] = lastgaelam = delta + ppo_args.gamma * ppo_args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

            self.returns = returns
            self.advantages = advantages

        if "tasks_results_buf" in infos:
            tasks_results_buf = infos["tasks_results_buf"]
        else:
            tasks_results_buf = None

        return global_step, next_obs, next_done, tasks_results_buf

    def update(self, global_step, update_i, tasks_results_buf):
        print("update_i", update_i)

        args = self.args
        ppo_args = self.ppo_args
        writer = self.writer

        # flatten the batch
        b_obs = self.obs.view(-1, self.obs.shape[-1])
        b_logprobs = self.logprobs.view(-1)
        b_actions = self.actions.view(-1, self.actions.shape[-1])
        b_advantages = self.advantages.view(-1)
        b_returns = self.returns.view(-1)
        b_values = self.values.view(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        initial_entropy = ppo_args.entropy_coef
        decay_rate = ppo_args.entropy_decay_rate
        entropy_coef = max(ppo_args.min_entropy, initial_entropy * (decay_rate ** update_i))

        for epoch in range(ppo_args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size,
                               self.minibatch_size):  # chop horizon length into batches, update with mini horizon length, unroll num_envs
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue, mu_hat = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                           b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds].reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > ppo_args.e_clip).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if ppo_args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # Policy loss
                pg_loss1 = -mb_advantages.reshape(-1) * ratio
                pg_loss2 = -mb_advantages.reshape(-1) * torch.clamp(ratio, 1 - ppo_args.e_clip, 1 + ppo_args.e_clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value lossf
        newvalue = newvalue.view(-1)
        if ppo_args.clip_value:
            v_loss_unclipped = (newvalue - b_returns[mb_inds].reshape(-1)) ** 2
            v_clipped = b_values[mb_inds].reshape(-1) + torch.clamp(
                newvalue - b_values[mb_inds].reshape(-1),
                -ppo_args.e_clip,
                ppo_args.e_clip,
            )
            v_loss_clipped = (v_clipped - b_returns[mb_inds].reshape(-1)) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds].reshape(-1)) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - entropy_coef * entropy_loss + v_loss * ppo_args.vf_coef

        if "UP_OSI" in args.network:
            mu = b_obs[mb_inds][:, -self.env._task.dr_size:]
            mu_loss = self.MSE_loss(mu, mu_hat)
            loss = mu_loss + loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), ppo_args.grad_norm)
        self.optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # print stats
        n_collision = 0
        n_reach_goal = 0
        n_timeout = 0
        if tasks_results_buf is not None:
            for idx in range(len(tasks_results_buf)):
                n_collision += tasks_results_buf[idx].count("collision")
                n_reach_goal += tasks_results_buf[idx].count("reach_goal")
                n_timeout += tasks_results_buf[idx].count("time_out")
            total_n = n_collision + n_reach_goal + n_timeout
            self.writer.add_scalar("Agent/Success_rate", n_reach_goal / total_n, global_step)
            self.writer.add_scalar("Agent/Timeout_rate", n_timeout / total_n, global_step)
            self.writer.add_scalar("Agent/Collision_rate", n_collision / total_n, global_step)

        sum_rewards = 0
        sum_length = 0
        for idx in range(len(self.ep_length_buf)):
            sum_rewards += self.ep_rewards_buf[idx]
            sum_length += self.ep_length_buf[idx]

        if len(self.ep_rewards_buf) > 0:
            self.writer.add_scalar('Episode_returns/Agent', sum_rewards / len(self.ep_rewards_buf),
                                   global_step)
            self.writer.add_scalar('Episode_length/Agent', sum_length / len(self.ep_length_buf),
                                   global_step)
            self.ep_rewards_buf = []
            self.ep_length_buf = []

            # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        if "UP_OSI" in args.network:
            writer.add_scalar("losses/mu_loss", mu_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/entropy_coef", entropy_coef, global_step)

        fps = int(global_step / (time.time() - self.start_time))
        writer.add_scalar("zothers/FPS", fps, global_step)
        if update_i % ppo_args.ppo_print_interval == 0 or update_i == self.num_updates:
            print("Step %d / %d:, Average returns: %.2f, with %d FPS" % (
                update_i, self.num_updates, self.reward_sum / (ppo_args.horizon_length * ppo_args.ppo_print_interval),
                fps))
            self.reward_sum = 0

    def eval(self, env, model_path=None, agent=None, global_step=0):
        # env._my_world.set_simulation_dt(1.0/env.args.eval_physics_dt)
        print("evaluation")
        if model_path is not None:
            eval_agent = Baseline_Agent(env).to(self.device)
            eval_agent.load_state_dict(torch.load(model_path))
        else:
            eval_agent = agent

        args = self.args
        ppo_args = self.ppo_args
        device = self.device

        obs = env.reset()
        eval_ep = 0
        ep_rewards_buf = np.zeros((args.num_envs,), dtype=object)
        ep_length_buf = np.zeros((args.num_envs,), dtype=object)
        ep_rewards = np.zeros((args.num_envs,), dtype=object)

        for i in range(args.num_envs):
            ep_rewards[i] = []
            ep_length_buf[i] = []
            ep_rewards_buf[i] = []

        total_ep_rew = 0
        env._task.eval_mode = True

        while eval_ep < int(ppo_args.eval_episodes) * args.num_envs:

            actions = eval_agent.get_eval(torch.Tensor(obs['obs']).to(self.device))
            # actions = actions.detach().cpu().numpy()

            next_obs, rewards, dones, infos = env.step(actions)
            for i in range(args.num_envs):
                ep_rewards[i].append(copy.deepcopy(rewards[i]))
                if dones[i] and len(ep_length_buf[i]) < ppo_args.eval_episodes:
                    ep_rewards_buf[i].append(sum(ep_rewards[i]))
                    ep_length_buf[i].append(len(ep_rewards[i]))
                    # ep_info = {"r": round(ep_rew, 6), "l": ep_len}
                    eval_ep += 1
                    print("eval ep:", eval_ep)

                    self.ep_rewards[i] = []
            else:
                obs = next_obs

        print("eval record tensorboard")
        n_collision = 0
        n_reach_goal = 0
        n_timeout = 0
        sum_rewards = 0
        sum_length = 0
        total_n = 0
        total_ep = 0

        if "eval_tasks_results_buf" in infos:
            for idx in range(len(infos["eval_tasks_results_buf"])):
                n_collision += infos["eval_tasks_results_buf"][idx].count("collision")
                n_reach_goal += infos["eval_tasks_results_buf"][idx].count("reach_goal")
                n_timeout += infos["eval_tasks_results_buf"][idx].count("time_out")
                sum_rewards += sum(ep_rewards_buf[idx])
                sum_length += sum(ep_length_buf[idx])
                total_ep += len(ep_length_buf[idx])
            total_n = n_collision + n_reach_goal + n_timeout

        if total_n > 0:
            self.writer.add_scalar("Eval_Agent/Success_rate", n_reach_goal / total_n, self.total_eval_ep)
            self.writer.add_scalar("Eval_Agent/Timeout_rate", n_timeout / total_n, self.total_eval_ep)
            self.writer.add_scalar("Eval_Agent/Collision_rate", n_collision / total_n, self.total_eval_ep)
            print("eval Success rate: ", n_reach_goal / total_n)

        if total_ep > 0:
            self.writer.add_scalar('Eval_Agent/Episode_returns', sum_rewards / total_ep,
                                   self.total_eval_ep)
            self.writer.add_scalar('Eval_Agent/Episode_length', sum_length / total_ep,
                                   self.total_eval_ep)
            print("eval ep rewards: ", sum_rewards / total_ep)
        self.total_eval_ep += 1
        env._task.eval_mode = False

        return total_ep_rew / ppo_args.eval_episodes
