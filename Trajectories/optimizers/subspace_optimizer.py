from utils.config_space_analysis import make_subspaces_by_conditions
import ConfigSpace
import copy

class SubspaceOptimizer:
    def __init__(
        self,
        space,
        subsapce_optimizer,
        subsapce_optimizer_parameters,
        sub_space_num,
        seed,
        initial_configs=None,
        limit_to_configs=None,
    ):
        self.space = space
        self.subsapce_optimizer = subsapce_optimizer
        self.subsapce_optimizer_parameters = subsapce_optimizer_parameters
        self.metrics = [subsapce_optimizer_parameters["metrics"]]
        self.bucket = subsapce_optimizer_parameters["bucket"]
        self.seed = seed
        self.t = 0
        self.subspace_optimizers = []
        self.subsapces, self.arms_names, self.parent_names = (
            make_subspaces_by_conditions(space)
        )
        self.number_of_arms = len(self.arms_names)

        self.initial_configs = initial_configs
        self.limit_to_configs = limit_to_configs
        self.sub_space_num = sub_space_num

        sub_space = self.subsapces[sub_space_num]
        if "n_configs" in self.subsapce_optimizer_parameters:
            if self.subsapce_optimizer_parameters["n_configs"] is None:
                self.subsapce_optimizer_parameters["n_configs"] = (
                    self.subsapce_optimizer_parameters["n_trials"]
                    // (self.number_of_arms * 4)
                )

        if self.initial_configs is not None:
            config = dict(self.initial_configs[sub_space_num])
            del config[self.parent_names[sub_space_num]]
            initial_configs = [
                ConfigSpace.configuration_space.Configuration(sub_space, values=config)
            ]
            if "n_configs" in self.subsapce_optimizer_parameters:
                self.subsapce_optimizer_parameters["n_configs"] -= 1

        if self.limit_to_configs is not None:
            limit_to_configs = []
            for config in self.limit_to_configs:
                config = dict(config)
                if self.parent_names[sub_space_num] in config:
                    if (
                        config[self.parent_names[sub_space_num]]
                        == self.arms_names[sub_space_num]
                    ):
                        new_config = copy.deepcopy(config)
                        del new_config[self.parent_names[sub_space_num]]
                        new_config = ConfigSpace.configuration_space.Configuration(
                            sub_space, values=new_config
                        )
                        limit_to_configs.append(new_config)

        self.subspace_optimizer = self.subsapce_optimizer(
            space=sub_space,
            **self.subsapce_optimizer_parameters,
            initial_configs=initial_configs,
            limit_to_configs=limit_to_configs,
        )
        self.last_trial_name = None

    def ask(self):
        trial = self.subspace_optimizer.ask()
        trial = copy.deepcopy(trial)
        self.last_trial_name = trial.name
        trial.name = (
            "exp"
            + str(self.t)
            + "-"
            + self.arms_names[self.sub_space_num]
            + "-"
            + trial.name
        )
        trial.config[self.parent_names[self.sub_space_num]] = self.arms_names[
            self.sub_space_num
        ]
        return trial

    def tell(self, report):
        report = copy.deepcopy(report)
        arm_name = report.trial.name.split("-")[1]
        self.selected_arm = self.arms_names.index(arm_name)
        report.trial.name = report.trial.name.split("-")[-1]
        del report.trial.config[self.parent_names[self.selected_arm]]
        self.subspace_optimizer.tell(report)
        self.t += 1
