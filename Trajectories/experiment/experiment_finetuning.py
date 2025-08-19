import os
from amltk.store import PathBucket
from amltk.optimization import History
import pickle
from amltk.optimization import Trial
from amltk.store import Stored
import numpy as np
from amltk.optimization import Metric
from experiment.dataset import get_data_splits
from optimizers.dummy import DefaultConfiguration
from tabularlab import Model
import copy
import psutil

def optimization_with_finetuning(
    base_output_path,
    base_model_name,
    finetuning_type,
    iterations,
    time_limit,
    dataset,
    task_id,
    save_history_freq,
    log_output_path,
    worker_info,
    seed,
):
    available_cores, n_worker_task, device = worker_info
    p = psutil.Process()
    p.cpu_affinity(available_cores[seed * n_worker_task:(seed+1)*n_worker_task]) 
    print(f"Process {p.pid} seed {seed} running on cores: {p.cpu_affinity()}, n_worker_task={n_worker_task}")
    print(f"Using device: {device}")

    results = {}
    trial_name = str(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    results["random_seed"] = seed
    results["base_model_name"] = base_model_name
    results["optimization_type"] = finetuning_type

    output_path = base_output_path.joinpath(trial_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    log_output_path_trial = log_output_path.joinpath(trial_name)
    bucket = PathBucket(log_output_path_trial, clean=False, create=True)

    (X, y), dataset_info = get_data_splits(dataset, rng)

    model = Model(X, y, dataset_info["task_type"], base_model_name, random_state=seed, device=device)

    metric = Metric("loss", minimize=True)
    optimizer = DefaultConfiguration(metric, model.get_hyperparameter_search_space(), bucket, seed, finetuning_type)
    remaining_iterations = iterations
    history = History()
    # if os.path.exists(output_path.joinpath("history.pkl")):
    #     print("Resuming!")
    #     with open(output_path.joinpath("history.pkl"), "rb") as handle:
    #         history = pickle.load(handle)

    #     if iterations <= len(history):
    #         print("all results are ready")
    #         return output_path  # exit()
    if base_model_name == "tabforestpfn":
        best_model = model
        best_weights = copy.deepcopy(model.model.model.model.state_dict())
    else:
        best_model = copy.deepcopy(model.model)
        best_weights = None

    best_loss = np.inf

    print(str(remaining_iterations) + " iterations is left")

    try:
        for iteration in range(1, remaining_iterations + 1):
            trial = optimizer.ask()
            report, internal_vall_loss = evaluate(
                trial=trial,
                model=model,
                X=X,
                y=y,
                dataset_info=dataset_info,
                best_model=best_model,
                best_loss=best_loss,
                base_model_name=base_model_name,
                best_weights =best_weights
            )
            if(best_loss > internal_vall_loss):
                best_loss = internal_vall_loss
            history.add(report)
            optimizer.tell(report)
            if iteration % save_history_freq == 0:
                with open(output_path.joinpath("history.pkl"), "wb") as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Loop stopped due to an error: {e}")

    if "ptarl" in base_model_name:
        for item in history:
            item.config["d_layers"] = "["+' '.join(str(x) for x in item.config["d_layers"]) +"]"
    history_df = history.df()
    history_df.to_pickle(output_path.joinpath("result.pkl"))
    return output_path


def evaluate(
    trial: Trial,
    model,
    X: Stored[np.ndarray],
    y: Stored[np.ndarray],
    dataset_info,
    best_model,
    best_loss,
    base_model_name,
    best_weights = None
) -> Trial.Report:
    with trial.profile("cross-validate"):
        train_indices = list(
            map(
                dataset_info["k_fold_split_indx"].__getitem__,
                dataset_info["train_folds"],
            )
        )
        fold = 0
        internal_val = train_indices[fold]
        internal_train = train_indices[:fold] + train_indices[fold + 1 :]
        X_internal_train = X.iloc[np.concatenate(internal_train)]
        y_internal_train = y.iloc[np.concatenate(internal_train)]
        X_internal_val = X.iloc[internal_val]
        y_internal_val = y.iloc[internal_val]

        with trial.profile("fit"):
            model.partial_fit(X_internal_train, y_internal_train)

        with trial.profile("predictions"):
            internal_val_loss, pred = model.eval(X_internal_val, y_internal_val)
            if base_model_name == "tabforestpfn":
                if best_loss <= internal_val_loss:
                    old_weights = copy.deepcopy(model.model.model.model.state_dict())
                    model.model.model.model.load_state_dict(best_weights)
                else:
                    best_weights = copy.deepcopy(model.model.model.model.state_dict())
                best_model = model
                    
            else:
                if best_loss > internal_val_loss:
                        best_model = copy.deepcopy(model.model)

        with trial.profile("scoring"):
            for i, fold in enumerate(dataset_info["val_folds"]):
                X_val = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                y_val = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                loss, pred = best_model.eval(X_val, y_val)
                trial.summary["val_loss" + str(i)] = loss

            for i, fold in enumerate(dataset_info["test_folds"]):
                X_test = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                y_test = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                loss, pred = best_model.eval(X_test, y_test)
                trial.summary["test_loss" + str(i)] = loss
                
            trial.summary["internal_val_loss"] = internal_val_loss

    if base_model_name == "tabforestpfn":
        if best_loss <= internal_val_loss:
            model.model.model.model.load_state_dict(old_weights)

    return trial.success(loss=internal_val_loss), internal_val_loss