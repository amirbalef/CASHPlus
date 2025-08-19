import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Append the current working directory
import lcpfn
from .utils import reward_shaper, target_f_t

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", device)
else:
    device = torch.device("cpu")
    print("No GPU -> using CPU:", device)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class PS_PFNs():
    def __init__(
        self,
        narms,
        T=200,
        alpha=0.1,
        max_budget=1000,
        models_per_arm=None,
        max_target="current_steps",
    ):
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
        self.max_budget = max_budget
        self.pfn_max_iteration = 199
        self.models = []
        self.points = []

        models_path = {
            "curved": "../Analysis/PFN/trained_models/paper curved_04-24-16-00-13_full_model.pt",
            "semi_flat": "../Analysis/PFN/trained_models/paper semi-Flat_04-24-16-00-12_full_model.pt",
            "flat": "../Analysis/PFN/trained_models/paper Flat_04-24-16-00-05_full_model.pt",
        }

        for arm, model_name in enumerate(models_per_arm):
            self.models.append(
                lcpfn.LCPFN(model_name=models_path[model_name]).to(device)
            )

            self.points.append(
                (
                    (
                        self.models[arm].model.criterion.borders[:-1]
                        + self.models[arm].model.criterion.bucket_widths / 2
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
            )
        self.PMFs = np.zeros((self.narms, len(self.points[0])))
        self.max_target = max_target

    def update_pmf(self, rewards, arm, iteration):
        n = len(rewards)
        if n > self.pfn_max_iteration:
            n = self.pfn_max_iteration
            acc_max = np.maximum.accumulate(rewards)[-self.pfn_max_iteration :]
        else:
            acc_max = np.maximum.accumulate(rewards)
            
        xs = np.arange(1, n + 1 + 1).reshape(n + 1, 1, 1)
        xs[-1, 0, 0] = iteration + 1
        ys = np.zeros((n + 1, 1, 1))
        ys[:n, 0, 0] = acc_max
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        with torch.no_grad():
            logits = self.models[arm](
                x_train=xs[:n, 0].to(device),
                y_train=ys[:n, 0].to(device),
                x_test=xs[-1:, 0].to(device),
            )
        predictions = torch.softmax(logits[:, :, :, 0], -1)
        self.PMFs[arm] = predictions.detach().cpu().numpy()[-1, 0, :]

    def policy_func(self):
        policy = np.zeros(self.narms)
        for arm in range(self.narms):
            if self.avaiable_arms[arm] == 1:
                reward_seq = np.asarray(
                    self.rewards[np.asarray(self.pulled_arms) == arm]
                )[:, 0]
                n = len(reward_seq)
                f_t = target_f_t(n , self.t, self.T, max_target = self.max_target)
                self.update_pmf(
                    reward_seq,
                    arm,
                    iteration=min( f_t , self.pfn_max_iteration),
                )
                cdf = np.cumsum(self.PMFs[arm])
                sample = np.searchsorted(cdf, np.random.uniform(0.01, 0.99))
                policy[arm] = self.points[arm][sample]
            else:
                policy[arm] = 0
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

    def update_loss(self, loss, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-loss)
        self.rewards = reward_shaper(self.raw_rewards, self.narms )
        self.t += 1