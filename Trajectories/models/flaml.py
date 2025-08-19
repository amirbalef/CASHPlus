from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from flaml import AutoML


class FLAML:
    def __init__(self, X, y, task_type, random_state = 0, output_path=None, num_cpus = 1, num_gpus = 0, **hyperparameters):
        self.automl_experiment = AutoML()
        self.task_type = task_type
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        task = (
            "classification"
            if task_type == "binary" or task_type == "multiclass"
            else "regression"
        )
        self.automl_settings = {
            "task": task,
            "seed": random_state,
            "n_jobs": self.num_cpus,
            "model_history": True,
            "log_file_name": output_path.joinpath("flaml.log"),
        }

    def partial_fit(self, X, y, time_limit=60):
        self.automl_settings["time_budget"] = time_limit
        self.automl_experiment.fit(X_train=X, y_train=y, **self.automl_settings)
        self.automl_settings["starting_points"] = (
            self.automl_experiment.best_config_per_estimator
        )
        return self

    def predict(self, X):
        return self.automl_experiment.predict(X)

    def eval(self, X, y):
        y_proba = self.predict_proba(X)
        if self.task_type == "binary":
            loss = 1 - roc_auc_score(y, y_proba[:, 1]) if y_proba is not None else None
        elif self.task_type == "multiclass":
            loss = log_loss(y, y_proba) if y_proba is not None else None
        else:
            y_preds = self.preprocess.y_encoder.inverse_transform(y_proba)
            loss = mean_squared_error(y, y_preds, squared=False)
        return loss, y_proba

    def predict_proba(self, X):
        if hasattr(self.automl_experiment, "predict_proba"):
            return self.automl_experiment.predict_proba(X)
        return None

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        sample_size = Constant("model_type", value="automl")
        cs.add_hyperparameters([sample_size])
        return cs