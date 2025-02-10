import time
import torch
from cleanRL.utils import set_random_seed
from abc import ABC, abstractmethod


class Trainer(ABC):

    def __init__(self, env, args, batch_path, writer, debug, device):
        set_random_seed(args.seed)
        self.env = env
        self.args = args
        self.batch_path = batch_path
        self.debug = debug

        self.device = torch.device(device) #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.writer = writer
        self.start_time = time.time()
        self.reward_sum = 0
        self.agent = None

    @abstractmethod
    def update(self, global_step):
        pass

    @abstractmethod
    def rollout(self, global_step, obs):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self, env, model_path, global_step=0):
        pass
