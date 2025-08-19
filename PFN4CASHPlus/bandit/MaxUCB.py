import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class MaxUCB():
    def __init__(self, narms, T=200, alpha=0.25, intial_steps=1):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.intial_steps = intial_steps
        self.random_init = False
        self.alpha = alpha
        self.T = T

    def policy_func(self ):
        policy = np.zeros(self.narms)
        for i  in range(self.narms):
            reward_seq =  np.asarray(self.rewards[ np.asarray(self.pulled_arms) == i ])[:,0]

            max_X = np.max(reward_seq)

            pad = (self.alpha *np.log(self.t)/len(reward_seq))**2

            policy[i] = max_X + pad 
        return policy

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
        else:
            policy =  self.policy_func()
            self.selected_arm =  np.argmax(policy)  
        return self.selected_arm
    
    def update_loss(self, loss, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        reward = max(0, 1 - loss)
        self.raw_rewards.append(reward)
        self.rewards = np.asarray(self.raw_rewards).reshape(-1, 1)
        self.t += 1