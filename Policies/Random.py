import numpy as np 

class Random():
    def __init__(self, narms, T =200, max_budget = 1000):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.selected_arm = 0
        self.intial_steps = 1
        self.random_init = False
        self.avaiable_arms = np.ones(narms)

    def set_arm_unavailable(self, arm):
        self.avaiable_arms[arm] = 0
        
    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
        else:
            if(self.avaiable_arms.sum() == 0):
                self.selected_arm = (self.t - 1) % self.narms
            else:
                self.selected_arm = np.random.choice(np.where(self.avaiable_arms > 0)[0])
        return self.selected_arm

    def update_loss_and_cost(self, loss, cost =None, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.t += 1

    def update_loss(self, loss, arm=None, context=None):
        if arm ==None:
            arm = self.selected_arm 
        self.pulled_arms.append(arm)
        self.t += 1
