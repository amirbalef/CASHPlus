from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss
from autogluon.tabular import TabularPredictor
import pandas as pd


class AutoGluon:
    def __init__(self, X, y, task_type, random_state = 0, output_path=None, num_cpus = 1, num_gpus = 0 , **hyperparameters):
        self.output_path = output_path
        self.automl_experiment = TabularPredictor(
            label="class", problem_type=task_type, path=output_path, verbosity=0,
        )
        self.task_type = task_type
        self.is_fit = False
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

    def partial_fit(self, X, y, time_limit=60):
        df = pd.DataFrame(X)
        df["class"] = y
        if self.is_fit:
            self.automl_experiment = TabularPredictor.load(self.output_path)
            self.automl_experiment.fit_extra("default", time_limit=time_limit, num_cpus = self.num_cpus, num_gpus = self.num_gpus)
        else:
            self.automl_experiment.fit(
                df,
                time_limit=time_limit,
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                #presets="best_quality",
            )
            self.is_fit = True
        return self

    def predict(self, X):
        df = pd.DataFrame(X)
        return self.automl_experiment.predict(df).to_numpy()

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
        df = pd.DataFrame(X)
        if hasattr(self.automl_experiment, "predict_proba"):
            return self.automl_experiment.predict_proba(df).to_numpy()
        return None

    def get_hyperparameter_search_space(self):
        cs = ConfigurationSpace()
        model_type = Constant("model_type", value="automl")
        cs.add_hyperparameters([model_type])
        return cs