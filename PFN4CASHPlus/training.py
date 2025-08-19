import argparse
from datetime import datetime
import yaml
import os
import torch
import wandb
from functools import partial
import lcpfn
from lcpfn import my_bar_distribution
from lcpfn import utils


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", device)
else:
    device = torch.device("cpu")
    print("No GPU -> using CPU:", device)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument("--config_number", nargs="?", default=0, type=int)
args = parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)
    return configs

configs = load_config("./configs/config_" + str(args.config_number) + ".yaml")

if configs["data"]["train_priors"] == "synth_ablation":
    from data.synth_ablation import get_batch_func as train_get_batch_func

    train_get_batch_func = partial(
        train_get_batch_func,
        outputs=configs["data"]["outputs"],
        function_index=configs["data"]["function_index"],
    )

if configs["data"]["train_priors"] == "synth_bandithpo":
    from data.synth_bandithpo import get_batch_func as train_get_batch_func
    train_get_batch_func = partial(
        train_get_batch_func,
        outputs=configs["data"]["outputs"],
    )

if configs["data"]["train_priors"] == "synth_hpo_maxucb":
    from data.synth_hpo_maxucb import get_batch_func as train_get_batch_func

    train_get_batch_func = partial(
        train_get_batch_func,
        outputs=configs["data"]["outputs"],
    )

if configs["data"]["train_priors"] == "synth_mix":
    from data.synth_mix import get_batch_func as train_get_batch_func

    train_get_batch_func = partial(
        train_get_batch_func,
        outputs=configs["data"]["outputs"],
    )

if configs["data"]["train_priors"] == "synth_paper":
    from data.synth_paper import get_batch_func as train_get_batch_func

    train_get_batch_func = partial(
        train_get_batch_func,
        outputs=configs["data"]["outputs"],
        function_index=configs["data"]["function_index"],
    )


if configs["data"]["val_priors"] == "real_data":
    from data.tasks_from_real_data import (
        get_dataset,
        get_batch_func,
    )
    real_data = get_dataset()
    val_get_batch_func = partial(
        get_batch_func,
        real_data,
        outputs=configs["data"]["outputs"],
    )


if configs["model"]["single_eval_pos_gen"] == "weighted":
    single_eval_pos_gen = utils.get_weighted_single_eval_pos_sampler(
        configs["model"]["seq_len"])
else:
    single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(
                configs["model"]["seq_len"], min_len=1)

if configs["model"]["decoder"] == "cascaded":
    from lcpfn.decoders.decoders import cascaded_decoder
    decoder = cascaded_decoder
else:
    from lcpfn.decoders.decoders import default_decoder
    decoder = default_decoder



# capture a dictionary of hyperparameters with config
configs["device"] = device

configs["saving_name"] = configs["model"]["name"]  +"_"+ datetime.now().strftime("%m-%d-%H-%M-%S") + ".pt"

print(configs)

if(args.config_number==0):  #config 0 is only for local tests
    wandb.init(mode="disabled")
    # wandb.init(
    #     project="PFN_HPO_task_v2",
    #     name=configs["model"]["name"],
    #     notes=configs["note"],
    #     config=configs,
    # )
else:
    # start a new experiment
    wandb.init(project="PFN_HPO_task_v4", name= configs["model"]["name"] , notes =configs["note"] , config=configs)

result = lcpfn.train_lcpfn(
    get_batch_func=train_get_batch_func,
    single_eval_pos_gen=single_eval_pos_gen,
    seq_len=configs["model"]["seq_len"],
    emsize=configs["model"]["emsize"],
    nlayers=configs["model"]["nlayers"],
    num_borders=configs["model"]["num_borders"],
    lr=configs["train"]["learning_rate"],
    batch_size=configs["train"]["batch_size"],
    epochs=configs["train"]["epochs"],
    num_features=1,
    outputs=configs["data"]["outputs"],
    num_outputs=len(configs["data"]["outputs"]),
    saving_period=configs["train"]["saving_period"],
    saving_name=configs["saving_name"],
    bar_distribution=my_bar_distribution,
    get_batch_func_val=val_get_batch_func,
    val_num_outputs=configs["data"]["val_num_output"],
    validation_period=1,
    decoder=decoder,
    borders_range=None
    if "borders_range" not in configs["model"].keys()
    else configs["model"]["borders_range"],
    max_seq_len=configs["data"]["max_seq_len"],
    weighted_loss=configs["train"]["weighted_loss"],
)