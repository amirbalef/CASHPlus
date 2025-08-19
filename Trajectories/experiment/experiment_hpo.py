import os
from amltk.store import PathBucket
from amltk.optimization import History
import pickle
from amltk.pipeline import Component
from tabularlab import Model
from amltk.optimization import Trial
from amltk.store import Stored
from amltk.pipeline import Sequential
import numpy as np
from optimizers.SMAC import SMACOptimizer
from amltk.optimization import Metric
from experiment.dataset import get_data_splits
import psutil


def get_pipeline(X, y, task_type, random_state, model_name, device):
    return Component(
        Model,
        space=Model.get_model_space(model_name, task_type),
        config={
            "X": X,
            "y": y,
            "task_type": task_type,
            "random_state": random_state,
            "model_name": model_name,
            "device": device,
        },
    )

def get_optimizer(hpo_name, metric, pipline, bucket, seed, initial_configs, n_trials):
    optimizer = SMACOptimizer.create(
        space=pipline,  # Let it know what to optimize
        metrics=metric,  # And let it know what to expect
        bucket=bucket,  # And where to store artifacts for trials and optimizer output
        seed=seed,
        initial_configs=initial_configs,
        n_trials=n_trials,
    )
    return optimizer


def evaluate(
    trial: Trial,
    pipeline: Sequential,
    X: Stored[np.ndarray],
    y: Stored[np.ndarray],
    dataset_info,
) -> Trial.Report:
    configured_pipeline = pipeline.configure(trial.config)
    internal_losses = []
    val_losses = []
    test_losses = []
    with trial.profile("cross-validate"):
        train_indices = list(map(dataset_info["k_fold_split_indx"].__getitem__, dataset_info["train_folds"]))
        for fold in range(3):
            internal_val = train_indices[fold]
            internal_train = train_indices[:fold] + train_indices[fold + 1 :]
            X_internal_train = X.iloc[np.concatenate(internal_train)]
            y_internal_train = y.iloc[np.concatenate(internal_train)]
            X_internal_val = X.iloc[internal_val]
            y_internal_val = y.iloc[internal_val]

            the_pipeline = configured_pipeline.build("sklearn")
            with trial.profile("fit"):
                the_pipeline.fit(X_internal_train, y_internal_train)

            with trial.profile("predictions"):
                loss, pred = the_pipeline._final_estimator.eval(
                    X_internal_val, y_internal_val
                )
                internal_losses.append(loss)

            with trial.profile("scoring"):
                losses = []
                for i, fold in enumerate(dataset_info["val_folds"]):
                    X_val = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                    y_val = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                    loss, pred = the_pipeline._final_estimator.eval(X_val, y_val)
                    losses.append(loss)
                val_losses.append(losses)

                losses = []
                for i, fold in enumerate(dataset_info["test_folds"]):
                    X_test = X.iloc[dataset_info["k_fold_split_indx"][fold]]
                    y_test = y.iloc[dataset_info["k_fold_split_indx"][fold]]
                    loss, pred = the_pipeline._final_estimator.eval(X_test, y_test)
                    losses.append(loss)
                test_losses.append(losses)

    internal_losses = np.mean(internal_losses)
    trial.summary["internal_val_loss"] = internal_losses

    for i, fold in enumerate(dataset_info["val_folds"]): 
        trial.summary["val_loss" + str(i)] = np.mean(val_losses, axis = 0)[i]

    for i, fold in enumerate(dataset_info["test_folds"]):
        trial.summary["test_loss" + str(i)] = np.mean(test_losses, axis = 0)[i]

    return trial.success(loss=internal_losses)


def optimization_with_hpo(
    base_output_path,
    base_model_name,
    hpo_name,
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
    results["hpo_name"] = hpo_name

    output_path = base_output_path.joinpath(trial_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_output_path_trial = log_output_path.joinpath(trial_name)
    bucket = PathBucket(log_output_path_trial, clean=False, create=True)

    (X, y), dataset_info = get_data_splits(dataset, rng)

    pipeline = get_pipeline(
        X, y, dataset_info["task_type"], seed, base_model_name, device
    )

    metric = Metric("loss", minimize=True)

    initial_configs = [pipeline.search_space(parser="configspace").get_default_configuration()]
    optimizer = get_optimizer(hpo_name, metric, pipeline, bucket, seed, initial_configs, min(iterations, 100))

    remaining_iterations = iterations
    history = History()
    # if os.path.exists(output_path.joinpath("history.pkl")):
    #     print("Resuming!")
    #     with open(output_path.joinpath("history.pkl"), "rb") as handle:
    #         history = pickle.load(handle)

    #     if iterations <= len(history):
    #         print("all results are ready")
    #         return output_path  # exit()
    #     for report in history:
    #         #print(report)
    #         optimizer.tell(report)
    #         remaining_iterations -= 1
    print(str(remaining_iterations) + " iterations is left")
    try:
        for iteration in range(1, remaining_iterations + 1):
            print(iteration)
            trial = optimizer.ask()
            report = evaluate(
                trial=trial,
                pipeline=pipeline,
                X=X,
                y=y,
                dataset_info=dataset_info,
            )
            history.add(report)
            optimizer.tell(report)
            if iteration % save_history_freq == 0:
                with open(output_path.joinpath("history.pkl"), "wb") as handle:
                    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Loop stopped due to an error: {e}")

    history_df = history.df()
    history_df.to_pickle(output_path.joinpath("result.pkl"))
    return output_path
