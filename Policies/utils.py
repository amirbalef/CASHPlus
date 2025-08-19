import numpy as np
from scipy.stats import truncnorm
from scipy.stats import invgamma, norm

class CostEstimator():
    """
    A class to estimate the cost of a given policy.
    """

    def __init__(self, narms, max_budget, max_iterations):
        self.max_budget = max_budget
        self.max_cost = self.max_budget
        self.narms = narms
        self.oberserved_costs = [ [] for _ in range(narms)]
        self.lower_cost_factor = 0.5
        self.upper_cost_factor = 1.5

    def sample_from_cost_distribution(self, arm):
        """
        Estimate the cost of the given policy in the given state.
        """
        n = len(self.oberserved_costs[arm])
        # Parameters
        mu, sigma = 1, 1/n
        a, b = (self.lower_cost_factor - mu) / sigma, (self.upper_cost_factor - mu) / sigma
        sample_factor = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1)

        cost = self.spend_budget(arm) / n * sample_factor
        if cost < 0.1:
            cost = 0.1
        if cost > self.max_cost:
            cost = self.max_cost
        return cost

    def spend_budget(self, arm):
        """
        Get the spent budget for a given arm.
        """
        return np.sum(self.oberserved_costs[arm])
    def all_spend_budget(self):
        """
        Get the spent budget for all arms.
        """
        return sum([item for sublist in self.oberserved_costs for item in sublist])

    def update_cost(self, arm, cost):
        self.oberserved_costs[arm].append(cost)




class CostEstimator_log_nromal():
    """
    A class to estimate the cost of a given policy.
    """

    def __init__(self, narms, max_budget, max_iterations):
        self.max_budget = max_budget
        self.max_cost = self.max_budget / 10
        self.min_cost = 0.1
        self.narms = narms
        self.oberserved_costs = [[] for _ in range(narms)]
        self.mu_0 = 0.0
        self.lambda_0=1.0
        self.alpha_0=1.0
        self.beta_0 =0.0

    def _posterior_params(self, arm):
        y = np.log(np.asarray(self.oberserved_costs[arm]))
        n = len(self.oberserved_costs[arm])
        if n == 0:
            return self.mu_0, self.lambda_0, self.alpha_0, self.beta_0
        y_bar = np.mean(y)
        s2 = np.var(y, ddof=0) if n > 1 else 0.0  # use population variance
        lambda_n = self.lambda_0 + n
        mu_n = (self.lambda_0 * self.mu_0 + n * y_bar) / lambda_n
        alpha_n = self.alpha_0 + n / 2
        beta_n = self.beta_0 + 0.5 * (n * s2 + (self.lambda_0 * n * (y_bar - self.mu_0) ** 2) / lambda_n)
        return mu_n, lambda_n, alpha_n, beta_n

    def sample_from_cost_distribution(self, arm):
        """
        Estimate the cost of the given policy in the given state.
        """
        mu_n, lambda_n, alpha_n, beta_n = self._posterior_params(arm)
        sigma2 = invgamma.rvs(a=alpha_n, scale=beta_n)
        mu = norm.rvs(loc=mu_n, scale=np.sqrt(sigma2 / lambda_n))
        cost = np.exp(norm.rvs(loc=mu, scale=np.sqrt(sigma2)))
        if cost < self.min_cost:
            cost = self.min_cost
        if cost > self.max_cost:
            cost = self.max_cost
        return cost

    def spend_budget(self, arm):
        """
        Get the spent budget for a given arm.
        """
        return np.sum(self.oberserved_costs[arm])

    def all_spend_budget(self):
        """
        Get the spent budget for all arms.
        """
        return sum([item for sublist in self.oberserved_costs for item in sublist])

    def update_cost(self, arm, cost):
        self.oberserved_costs[arm].append(cost)


def target_f_b(n, spend_budget, cost, max_budget, max_target = "current_steps"):
    if max_target == "current_steps":
        return spend_budget/cost
    elif max_target == "remaining_steps":
        return  n + (max_budget- spend_budget)/cost
    else:
        raise ValueError("max_target must be 'current_step' or 'remaining_steps'")

def target_f_t(n , t, T, max_target = "current_steps"):
    if max_target == "current_steps":
        return t
    elif max_target == "remaining_steps":
        return  n + max(T - t + 1, 1)
    else:
        raise ValueError("max_target must be 'current_step' or 'remaining_steps'")




def reward_shaper(raw_rewards, number_of_arms=1, type="softmax", for_pfn=True):
    if type == "no_change":
        return np.asarray(raw_rewards).reshape(-1, 1)
    if type == "linear":
        reward = np.asarray(raw_rewards).reshape(-1, 1) 
        d = reward[:number_of_arms] 

        if  (-1 <= d).all():  # accuracy 
            reward = 1 + reward 
            d = 1 + d
            n_reward = (reward - d.min()) / (reward.max() + 0.05)
        else:
            n_reward = (reward -  d.min()) / (reward.max() -  reward.min() + 0.05)

        n_reward[n_reward < 0 ] = 0
        return n_reward
    if type == "minmax":
        reward = np.asarray(raw_rewards).reshape(-1, 1) 
        if for_pfn:
            n_reward = 0.25 + 0.5*(reward - reward.min()) / (reward.max() - reward.min() + 0.01)
        else:
            n_reward = (reward - reward.min()) / (reward.max() - reward.min() + 0.01)
        n_reward[n_reward < 0] = 0
        return n_reward

    if type == "softmax":
        reward = 1+ np.asarray(raw_rewards).reshape(-1, 1)
        d = reward[:number_of_arms]
        n_reward = 1 / (1 + np.exp(-(reward - np.mean(d)) / (np.std(d) +0.01)))
        return   n_reward