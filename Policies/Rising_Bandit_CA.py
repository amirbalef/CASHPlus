import numpy as np 
from .utils import reward_shaper
 
class Rising_Bandit_CA():
    def __init__(self, narms, T=200, C=7, max_budget=1000):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.S_cand = np.ones(self.narms, dtype=int)
        self.u_values = np.ones(self.narms)
        self.l_values = np.zeros(self.narms)
        self.C = C
        self.S_cand_arms_to_be_played = np.ones(self.narms, dtype=int)
        self.avaiable_arms = np.ones(narms)
        self.spent_budget = np.zeros(narms)
        self.max_budget = max_budget

    def set_arm_unavailable(self, arm):
        self.avaiable_arms[arm] = 0
        if np.sum(self.S_cand * self.avaiable_arms)== 0:
            self.S_cand = np.copy(self.avaiable_arms) 
            self.S_cand_arms_to_be_played = np.zeros(self.narms)
            self.process()

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms==0):
            self.selected_arm =  (self.t -1)%self.narms
        else:
            if np.sum(self.S_cand_arms_to_be_played * self.avaiable_arms ) == 0:
                self.selected_arm = np.where(self.avaiable_arms)[0][0]
            else:
                self.selected_arm = np.where(self.S_cand_arms_to_be_played * self.avaiable_arms)[0][0]
                
        return self.selected_arm

    def process(self):
        self.S_cand_arms_to_be_played[self.selected_arm] = 0
        if sum(self.S_cand_arms_to_be_played) == 0:
            if sum(self.S_cand) > 1:
                for x in range(self.narms):
                    if self.S_cand[x]:
                        indx = np.asarray(self.pulled_arms) == x
                        rewards = np.maximum.accumulate(self.rewards[indx])[:, 0]
                        if len(rewards) > 0:
                            y_t = rewards[-1]
                            y_t_C = (
                                rewards[-self.C - 1] if (len(rewards) > self.C) else -1
                            )
                            self.l_values[x] = y_t
                            omega = (y_t - y_t_C) / self.C
                            self.u_values[x] = min(
                                1, y_t + omega * len(rewards)*(self.max_budget - np.sum(self.spent_budget)/self.spent_budget[x])
                            )
                if sum(self.S_cand) > 0:
                    for i in range(self.narms):
                        if self.S_cand[i]:
                            for j in range(self.narms):
                                if self.S_cand[j] and i != j:
                                    if self.l_values[i] >= self.u_values[j]:
                                        self.S_cand[j] = 0
            self.S_cand_arms_to_be_played = np.array((self.S_cand == 1), dtype=int)

    def update_loss_and_cost(self, loss, cost=None, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-loss)
        self.spent_budget[arm] += cost
        self.rewards = reward_shaper(self.raw_rewards, self.narms)
        self.process()
        self.t += 1