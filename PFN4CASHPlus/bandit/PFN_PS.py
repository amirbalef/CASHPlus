import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def output_func(output, data):
    if output == "max":
        return np.maximum.accumulate(data)
    if output == "min":
        return np.minimum.accumulate(data)
    if output == "raw":
        return data


class PFN_PS:
    def __init__(self, narms, T=200, model=None, device=None, outputs=['max']):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.random_init = False
        self.intial_steps = 1
        self.T = T
        self.model = model
        self.device = device
        self.points = (
            (
                self.model.criterion.borders[:-1]
                + self.model.criterion.bucket_widths / 2
            )
            .detach()
            .cpu()
            .numpy()
        )
        self.PMFs = np.zeros((self.narms, len(self.points)))
        self.outputs = outputs

    def policy_func(self):
        policy = np.zeros(self.narms)
        for arm in range(self.narms):
            reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == arm])[:, 0]
            self.update_pmf(reward_seq, arm)
            cdf = np.cumsum(self.PMFs[arm])
            policy[arm] = self.points[np.searchsorted(cdf, np.random.rand())]
        return policy

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)

        if np.any(pulled_arms < self.intial_steps):
            if self.random_init:
                self.selected_arm = np.random.choice(
                    np.where(pulled_arms < self.intial_steps)[0]
                )
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            policy = self.policy_func()
            self.selected_arm = np.random.choice(np.where(policy == policy.max())[0])
        return self.selected_arm

    def update_pmf(self, rewards, arm):
        n = len(rewards)
        xs = np.arange(1, n +1 + 1).reshape(n +1, 1, 1)
        xs [-1, 0, 0] = self.T - self.t + 1
        ys = np.zeros((n + 1, 1, len(self.outputs)))
        for o, output in enumerate(self.outputs):
            ys[:n, 0, o] = output_func(output, rewards)
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        with torch.no_grad():
            logits = self.model((xs.to(self.device), ys.to(self.device)), single_eval_pos=n)
        predictions = torch.softmax(logits[:, :, :, 0], -1)
        self.PMFs[arm] = predictions.detach().cpu().numpy()[-1, 0, :]

    def update_loss(self, loss, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        reward = max(0, 1 - loss)
        self.raw_rewards.append(reward)
        self.rewards = np.asarray(self.raw_rewards).reshape(-1, 1)
        self.t += 1