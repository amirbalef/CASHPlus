from __future__ import annotations

from functools import partial
from pathlib import Path
import argparse

from amltk.scheduling import Scheduler
from experiment import experiment_hpo, experiment_finetuning, experiment_automl, experiment_phe
from experiment.dataset import get_data
import psutil

# python main_run_hpo.py --dataset TabRepo --instance arcene --optimizer RandomSearch_Arm_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="?", default="openml", help="dataset name")
    parser.add_argument("--device", nargs="?", default="cpu", help="can be  cuda or cpu")
    parser.add_argument(
        "--task_id",
        nargs="?",
        default=361110,
        type=int,
        help="instance of the dataset names",
    )
    parser.add_argument(
        "--time_limit", nargs="?", default=12*3600, type=int, help="for each iteration"
    )
    parser.add_argument(
        "--pipeline",
        nargs="?",
        default="tabpfn_v2_phe.phe",  # "autogluon.automl",  # "flaml.automl",
        help="",  # randomforest.smac #"xtab.finetuning",
    )

    parser.add_argument(
        "--iterations", nargs="?", default=1000, type=int, help="number_of_iterations"
    )
    parser.add_argument("--output_root_dir", nargs="?", default="results/")
    parser.add_argument("--output_log_dir", nargs="?", default="tmp/")
    parser.add_argument(
        "--trials", nargs="?", default=1, type=int, help="number of trials"
    )
    parser.add_argument(
        "--start_trials", nargs="?", default=0, type=int, help="number of trials"
    )

    parser.add_argument(
        "--n_worker_task",
        nargs="?",
        default=4,
        type=int,
        help="",
    )
    parser.add_argument(
        "--save_history_freq", nargs="?", default=1, type=int, help="save_history_freq"
    )

    args = parser.parse_args()
    print("args", args.__dict__)
    pipeline_name = str(args.pipeline)
    dataset_name = args.dataset
    device = args.device
    task_id = args.task_id
    iterations = args.iterations
    time_limit = args.time_limit
    save_history_freq = args.save_history_freq
    base_output_path = Path(
        args.output_root_dir
        + "/"
        + dataset_name
        + "/"
        + str(task_id)
        + "/"
        + pipeline_name
        + "/"
    )
    log_output_path = Path(
        args.output_log_dir
        + "/"
        + dataset_name
        + "/"
        + str(task_id)
        + "/"
        + pipeline_name
        + "/"
    )

    base_model_name, optimizer = pipeline_name.split(".")

    dataset = get_data(dataset_name, task_id)

    worker_info = psutil.Process().cpu_affinity(), args.n_worker_task , device
    print(f"Available CPU cores: {worker_info[0]} / {worker_info[1]}")
    print(f"Using device: {worker_info[2]}")

    if(optimizer in ["smac", "randomsearch"]):
        experiment_type = experiment_hpo.optimization_with_hpo
    elif(optimizer in ["finetuning"]):
        experiment_type = experiment_finetuning.optimization_with_finetuning
    elif optimizer in ["automl"]:
        experiment_type = experiment_automl.optimization_with_automl
    elif optimizer in ["phe"]:
        experiment_type = experiment_phe.optimization_with_phe
    else:
        NotImplementedError
        exit()
    
    experiment_task_per_seed = partial(
        experiment_type,
        base_output_path,
        base_model_name,
        optimizer,
        iterations,
        time_limit,
        dataset,
        task_id,
        save_history_freq,
        log_output_path,
        worker_info
    )

    scheduler = Scheduler.with_processes(max_workers=len(worker_info[0]))
    task = scheduler.task(experiment_task_per_seed)
    seed_numbers = iter(range(args.start_trials, args.start_trials + args.trials))
    
    results = []
    # When the scheduler starts, submit #n_worker_scheduler tasks to the processes
    @scheduler.on_start(repeat=args.trials)
    def on_start():
        n = next(seed_numbers)
        print(n)
        task.submit(n)

    # When the task is done, store the result
    @task.on_result
    def on_result(_, result: float):
        results.append(result)

    # Easy to incrementently add more functionallity
    @task.on_result
    def launch_next(_, result: float):
        if (n := next(seed_numbers, None)) is not None:
            task.submit(n)

    # React to issues when they happen
    @task.on_exception
    def stop_something_went_wrong(_, exception: Exception):
        scheduler.stop()

    @scheduler.on_timeout
    def stop_timeout():
        print("timeout")
        scheduler.stop()
        print("end")

    # Start the scheduler and run it as you like
    scheduler.run(timeout=time_limit, wait =False)
    print("Done, results are avaible in ", results)