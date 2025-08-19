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
import psutil


def generate_time_sequence(iterations, time_limit):
    time_blocks = [30, 60, 120, 300, 600, 1200, 3600, 4*3600]  # Time values in seconds
    repeats = [10, 10, 10, 10, 10, 10, 10]  # Repetitions for each value
    repeats.append( int(iterations - sum(repeats)))  # Add remaining iterations to the last value
    sequence = []
    for time, repeat in zip(time_blocks, repeats):
        sequence.extend([time] * repeat)
    return sequence


def optimization_with_automl(
    base_output_path,
    base_model_name,
    automl_type,
    iterations,
    time_limit,
    dataset,
    task_id,
    save_history_freq,
    log_output_path,
    worker_info,
    seed,
):
    available_cores, n_worker_task, device= worker_info
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
    results["optimization_type"] = automl_type

    output_path = base_output_path.joinpath(trial_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_output_path_trial = log_output_path.joinpath(trial_name)
    bucket = PathBucket(log_output_path_trial, clean=False, create=True)

    (X, y), dataset_info = get_data_splits(dataset, rng)

    if(base_model_name=="flaml"):
        from models.flaml import FLAML as Model
    if base_model_name == "autogluon":
        from models.autogluon import AutoGluon as Model
    else:
        NotImplementedError
    
    if device == "cuda":
        num_gpu = 1
    else:
        num_gpu = 0

    model = Model(X, y, dataset_info["task_type"], random_state=seed, output_path=log_output_path_trial, num_cpus= n_worker_task,  num_gpu = num_gpu)
    metric = Metric("loss", minimize=True)
    optimizer = DefaultConfiguration(
        metric, model.get_hyperparameter_search_space(), bucket, seed, automl_type
    )
    remaining_iterations = iterations
    history = History()
    # if os.path.exists(output_path.joinpath("history.pkl")):
    #     print("Resuming!")
    #     with open(output_path.joinpath("history.pkl"), "rb") as handle:
    #         history = pickle.load(handle)

    #     if iterations <= len(history):
    #         print("all results are ready")
    #         return output_path  # exit()
    print(str(remaining_iterations) + " iterations is left")
    time_slots = generate_time_sequence(iterations, time_limit)
    for iteration in range(1, remaining_iterations + 1):
        trial = optimizer.ask()
        report = evaluate(
            trial=trial,
            model=model,
            X=X,
            y=y,
            dataset_info=dataset_info,
            time_limit=time_slots[iteration - 1],
        )
        history.add(report)
        optimizer.tell(report)
        if iteration % save_history_freq == 0:
            with open(output_path.joinpath("history.pkl"), "wb") as handle:
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    history_df = history.df()

    history_df.to_pickle(output_path.joinpath("result.pkl"))
    return output_path


def evaluate(
    trial: Trial,
    model,
    X: Stored[np.ndarray],
    y: Stored[np.ndarray],
    dataset_info,
    time_limit,
) -> Trial.Report:
    with trial.profile("cross-validate"):
        train_indices = list(
            map(
                dataset_info["k_fold_split_indx"].__getitem__,
                dataset_info["train_folds"],
            )
        )
        X_internal_train = X.iloc[np.concatenate(train_indices)]
        y_internal_train = y.iloc[np.concatenate(train_indices)]
        with trial.profile("fit"):
            model.partial_fit(X_internal_train, y_internal_train, time_limit=time_limit)

        with trial.profile("predictions"):
            internal_val_loss, pred = model.eval(X_internal_train, y_internal_train)

        with trial.profile("scoring"):
            for i, fold in enumerate(dataset_info["val_folds"]):
                X_val = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                y_val = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                loss, pred = model.eval(X_val, y_val)
                trial.summary["val_loss" + str(i)] = loss

            for i, fold in enumerate(dataset_info["test_folds"]):
                X_test = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                y_test = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                loss, pred = model.eval(X_test, y_test)
                trial.summary["test_loss" + str(i)] = loss

            trial.summary["internal_val_loss"] = internal_val_loss

    return trial.success(loss=internal_val_loss)