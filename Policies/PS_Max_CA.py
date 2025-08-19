import numpy as np 
from scipy.stats import invgamma, norm
from .utils import CostEstimator_log_nromal, reward_shaper, target_f_b


class PS_Max_CA:
    def __init__(self, narms, T=200, alpha=0.1, max_budget=1000, max_target="current_steps"):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.alpha = alpha
        self.T = T
        self.intial_steps = 1
        self.avaiable_arms = np.ones(narms)
        self.mu_0 = 1.0
        self.lambda_0 = 1
        self.alpha_0 = 1
        self.beta_0 = 0.0

        self.max_budget = max_budget
        self.spent_budget = np.zeros(narms)
        self.cost_estimator = CostEstimator_log_nromal(narms, max_budget, T)
        self.max_target = max_target
        

    def policy_func(self):
        policy = np.zeros(self.narms)
        for arm in range(self.narms):
            reward_seq = np.asarray(self.rewards[np.asarray(self.pulled_arms) == arm])[
                :, 0
            ]
            n = len(reward_seq)
            mean_reward = np.mean(reward_seq)
            sum_squared_diffs = np.sum((reward_seq - mean_reward) ** 2)
            lambda_0 =self.lambda_0  + n
            mu_0 = (self.lambda_0) / lambda_0 * self.mu_0 + n / lambda_0 * mean_reward
            alpha_0 = self.alpha_0 + n / 2
            beta_0 = self.beta_0 + 0.5 * sum_squared_diffs + (n * (mean_reward -  mu_0 ) ** 2) / (2 *  lambda_0)
            sigma2_sample = invgamma.rvs(alpha_0, scale=beta_0)

            f_t = target_f_b(n, self.cost_estimator.all_spend_budget(), self.cost_estimator.sample_from_cost_distribution(arm), self.max_budget, max_target = self.max_target)
            p = np.random.uniform(0, 1)
            res = norm.ppf(p**(1/(f_t)), loc=mu_0, scale=np.sqrt(sigma2_sample / lambda_0))
            if res < 0:
                res = 0
            if res > 1:
                res = 1

            policy[arm] = self.avaiable_arms[arm] * res

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
            if self.avaiable_arms[self.selected_arm] == 0:
                self.selected_arm = np.random.choice(np.where(self.avaiable_arms == 1)[0])
        return self.selected_arm

    def update_loss_and_cost(self, loss, cost=None, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-loss)
        self.cost_estimator.update_cost(arm, cost)
        self.rewards = reward_shaper(self.raw_rewards, self.narms )
        self.t += 1


    def update_loss(self, loss, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-loss)
        self.rewards = reward_shaper(self.raw_rewards, self.narms )
        self.t += 1