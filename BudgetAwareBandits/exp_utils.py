import numpy as np
import pandas as pd
import torch 

def run_fake_expriment(alg, data_info, data, max_budget):
    number_of_arms = data_info["number_of_arms"]
    number_of_trails = data_info["number_of_trails"]

    result = []
    result_test = []
    result_pulled_arms = []
    for trial in range(number_of_trails):

        if alg == "Oracle_Arm":
            arm_mins = []
            for arm in range(number_of_arms):
                arm_data = data[(data["repetition"] == trial) & (data["arm_index"] == arm)]
                cumulative_eval_time = arm_data["eval_time"].cumsum()
                valid_indices = cumulative_eval_time[cumulative_eval_time <= max_budget].index
                arm_mins.append(arm_data.loc[valid_indices, "loss"].min())
            oracle_arm = np.argmin(arm_mins)
            arm = oracle_arm
        else:
            arm = -1

        pulled_arms = np.zeros(number_of_arms, dtype=int)
        np.random.seed(trial)
        result_loss = [np.nan]
        result_loss_test = [np.nan]
        result_budget_cost = [0]
        pulled_arms_list = []
        
        b = 0
        while b < max_budget:
            this_data = data[(data["arm_index"] == arm) & (data["repetition"] == trial)]
            if pulled_arms[arm] in this_data["iteration"].values:
                iteration = pulled_arms[arm]
            else:
                iteration = np.random.choice(this_data["iteration"].values)
            loss, eval_cost, loss_test = this_data.loc[(this_data["iteration"] == iteration),
                ["loss", "eval_time", "test_loss"],
            ].values[0]

            b += eval_cost
            pulled_arms[arm] += 1
            pulled_arms_list.append(arm)
            result_loss.append(loss)
            result_loss_test.append(loss_test)
            result_budget_cost.append(b / max_budget)

        result_timeseries = pd.Series(result_loss, index=result_budget_cost)
        result_timeseries_test = pd.Series(result_loss_test, index=result_budget_cost)
        pulled_arms_timeseries = pd.Series(
            pulled_arms_list, index=result_budget_cost[1:]
        )

        result.append(result_timeseries)
        result_test.append(result_timeseries_test)
        result_pulled_arms.append(pulled_arms_timeseries)
    return result, result_test, result_pulled_arms


from joblib import Parallel, delayed


def run_single_trial(alg, data, trial, number_of_arms, max_budget):
    pulled_arms = np.zeros(number_of_arms, dtype=int)
    np.random.seed(trial)
    torch.manual_seed(trial)
    torch.cuda.manual_seed(trial)
    torch.cuda.manual_seed_all(trial)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    policy = alg(number_of_arms, max_budget=max_budget)

    result_loss = [np.nan]
    result_loss_test = [np.nan]
    result_budget_cost = [0]
    pulled_arms_list = []
    b = 0

    while b < max_budget:
        arm = policy.play()
        iteration = pulled_arms[arm]

        try:
            loss, eval_cost, loss_test = data.loc[
                (data["arm_index"] == arm)
                & (data["repetition"] == trial)
                & (data["iteration"] == iteration),
                ["loss", "eval_time", "test_loss"],
            ].values[0]
        except IndexError:
            # Skip if data is missing
            break

        if hasattr(policy, "update_loss_and_cost"):
            policy.update_loss_and_cost(loss, eval_cost)
        else:
            policy.update_loss(loss)

        b += eval_cost
        pulled_arms[arm] += 1
        pulled_arms_list.append(arm)

        result_loss.append(loss)
        result_loss_test.append(loss_test)
        result_budget_cost.append(b / max_budget)

        if pulled_arms[arm] not in data[(data["arm_index"] == arm) & (data["repetition"] == trial)]["iteration"].values:
            if hasattr(policy, "set_arm_unavailable"):
                policy.set_arm_unavailable(arm)
            else:
                if hasattr(policy, "update_loss_and_cost"):
                    policy.update_loss_and_cost(0.0, 0.0)
                else:
                    policy.update_loss(0)

    result_timeseries = pd.Series(result_loss, index=result_budget_cost)
    result_timeseries_test = pd.Series(result_loss_test, index=result_budget_cost)
    pulled_arms_timeseries = pd.Series(pulled_arms_list, index=result_budget_cost[1:])

    return result_timeseries, result_timeseries_test, pulled_arms_timeseries


def run_expriment(alg, data_info, data, max_budget, n_jobs=1):
    number_of_arms = data_info["number_of_arms"]
    number_of_trails = data_info["number_of_trails"]

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_trial)(alg, data, trial, number_of_arms, max_budget)
        for trial in range(number_of_trails)
    )

    result, result_test, result_pulled_arms = zip(*results)
    return list(result), list(result_test), list(result_pulled_arms)
