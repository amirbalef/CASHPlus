import numpy as np 
from .utils import reward_shaper

class TS:
    def __init__(self, narms, T=200, max_budget=1000):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.T = T
        self.intial_steps = 1
        self.avaiable_arms = np.ones(narms)

    def policy_func(self):
        policy = np.zeros(self.narms)
        for i in range(self.narms):
            reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == i])[
                :, 0
                ]
            policy[i] =self.avaiable_arms[i]* np.random.normal( np.mean(reward_seq), 1 / len(reward_seq))

        return policy

    def set_arm_unavailable(self, arm):
        self.avaiable_arms[arm] = 0

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms < self.intial_steps):
            self.selected_arm = (self.t - 1) % self.narms
        else:
            policy = self.policy_func()
            self.selected_arm = np.argmax(policy)
        return self.selected_arm

    def update_loss(self, loss, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-loss)
        self.rewards = reward_shaper(self.raw_rewards, self.narms )
        self.t += 1