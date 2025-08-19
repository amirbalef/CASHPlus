
from amltk.optimization import Metric
from utils.config_space_analysis import make_initial_config
from optimizers.SMAC import SMACOptimizer
from optimizers.random_search import RandomSearch
from optimizers.bandit_optimizer import BanditOptimizer
from optimizers.subspace_optimizer import SubspaceOptimizer


import importlib

def get_optimizer(
    optimizer_method,
    pipeline,
    bucket,
    seed,
    limit_to_configs,
    iterations,
):
    metrics = Metric("model_error", minimize=True)

    space = pipeline.search_space("configspace")
    initial_configs = make_initial_config(space)

    if optimizer_method == "RandomSearch":
        optimizer = RandomSearch(
            space=space,
            metrics=metrics,
            bucket=bucket,
            seed=seed,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )
    elif optimizer_method == "SMAC":
        optimizer = SMACOptimizer.create(
            space=space,
            metrics=metrics,
            bucket=bucket,
            seed=seed,
            n_trials=iterations,
            initial_configs=initial_configs,
            n_configs=(iterations // 4) - len(initial_configs),
            limit_to_configs=limit_to_configs,
        )  #

    elif (
        "Bandit" in optimizer_method
    ):  # Bandit_suboptimzerName.suboptimzerParams_policyName.policyParams
        subsapce_optimizer_name = optimizer_method.split("_")[1].split(".")[0]
        subsapce_optimizer_parameters = {
            "metrics": metrics,
            "bucket": bucket,
            "seed": seed,
        }
        if subsapce_optimizer_name == "SMAC":
            subsapce_optimizer = SMACOptimizer.create
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
                "n_trials": iterations,
                "n_configs": None,
            }
        elif subsapce_optimizer_name == "RandomSearch":
            subsapce_optimizer = RandomSearch
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
            }

        policy_name = "_".join(optimizer_method.split("_")[2:]).split(".")[0]
        policy = getattr(
            importlib.import_module("optimizers.bandit_policies." + policy_name),
            policy_name,
        )
        policy_parameters = {}
        optimizer = BanditOptimizer(
            space=space,
            subsapce_optimizer=subsapce_optimizer,
            subsapce_optimizer_parameters=subsapce_optimizer_parameters,
            policy=policy,
            policy_parameters=policy_parameters,
            seed=seed,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )

    elif (
        "Arm" in optimizer_method
    ):  # SuboptimzerName.suboptimzerParams_Arm_arm-number
        subsapce_optimizer_name = optimizer_method.split("_")[0].split(".")[0]
        subsapce_optimizer_parameters = {
            "metrics": metrics,
            "bucket": bucket,
            "seed": seed,
        }
        if subsapce_optimizer_name == "SMAC":
            subsapce_optimizer = SMACOptimizer.create
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
                "n_trials": iterations,
                "n_configs": None,
            }
        elif subsapce_optimizer_name == "RandomSearch":
            subsapce_optimizer = RandomSearch
            subsapce_optimizer_parameters = {
                "metrics": metrics,
                "bucket": bucket,
                "seed": seed,
            }
        subspace_num = int(optimizer_method.split("_")[-1])
        optimizer = SubspaceOptimizer(
            space=space,
            subsapce_optimizer=subsapce_optimizer,
            subsapce_optimizer_parameters=subsapce_optimizer_parameters,
            sub_space_num=subspace_num,
            seed=seed,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )
    else:
        print("error")
        exit()

    return optimizer
