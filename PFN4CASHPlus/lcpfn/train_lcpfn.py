import math

import numpy as np
import torch

from torch import nn

from lcpfn import encoders, train
from lcpfn import utils

from lcpfn.priors import utils as putils
from lcpfn.decoders.decoders import default_decoder
from functools import partial

from torch import Tensor
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


def train_lcpfn(
    get_batch_func,
    single_eval_pos_gen,
    seq_len: int = 100,
    emsize: int = 512,
    nlayers: int = 12,
    num_borders: int = 1000,
    lr: float = 0.0001,
    batch_size: int = 100,
    epochs: int = 1000,
    num_features: int = 1,
    outputs: list = ["max"],
    num_outputs: int = 1,
    saving_period: int = 10,
    saving_name: str = "my_pfn.pt",
    get_batch_func_val=None,
    val_num_outputs=1,
    validation_period=1,
    bar_distribution=None,
    bar_distribution_type = "Full",
    decoder=default_decoder,
    borders_range=None,
    max_seq_len=200,
    weighted_loss=False,
):
    """
    Train a LCPFN model using the specified hyperparameters.

    Args:
        get_batch_func (callable): A function that returns a batch of learning curves.
        seq_len (int, optional): The length of the input sequence. Defaults to 100.
        emsize (int, optional): The size of the embedding layer. Defaults to 512.
        nlayers (int, optional): The number of layers in the model. Defaults to 12.
        num_borders_choices (int, optional): The number of borders to use. Defaults to 1000.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.0001.
        batch_size (int, optional): The batch size for training. Defaults to 100.
        epochs (int, optional): The number of epochs to train for. Defaults to 1000.

    Returns:
        torch.module: The trained model.
    """

    hps = {}

    # PFN training hyperparameters
    dataloader = putils.get_batch_to_dataloader(get_batch_func)  # type: ignore

    if borders_range == None:
        ys = get_batch_func(
            10_000,
            seq_len,
            num_features,
            hyperparameters=hps,
            single_eval_pos=seq_len,
        )
        print("borders_range=", ys[2].min(), ys[2].max())
        bucket_limits = bar_distribution.get_bucket_limits(num_borders, ys=ys[2])

    elif borders_range=="non_linear":
        loc = 0.1  # Location parameter
        scale = 0.3  # Scale parameter
        samples = 1 - np.random.exponential(scale, size=10_000* seq_len) + loc
        samples[samples<0] = 0.0
        samples[samples >1] = 1.0
        ys = torch.from_numpy(samples.reshape(seq_len, 10_000, 1).astype(np.float32))
        print("non_linear borders_range=", ys.min(), ys.max())
        bucket_limits = bar_distribution.get_bucket_limits(num_borders, ys=ys)

    else:
        bucket_limits = bar_distribution.get_bucket_limits(
            num_borders, full_range=borders_range
        )

    # Discretization of the predictive distributions
    if bar_distribution_type == "Full":
        criterions = {
            num_features: {
                num_borders: bar_distribution.FullSupportBarDistribution(bucket_limits)
            }
        }
    else:
        criterions = {
            num_features: {
                num_borders: bar_distribution.BarDistribution(bucket_limits)
            }
        }

    if(get_batch_func_val != None):
        val_dataloader = putils.get_batch_to_dataloader(get_batch_func_val)  # type: ignore
    else:
        val_dataloader = None

    config = dict(
        nlayers=nlayers,
        priordataloader_class=dataloader,
        criterion=criterions[num_features][num_borders],
        encoder_generator=lambda in_dim, out_dim: nn.Sequential(
            encoders.MinMax_Normalize(0.0, max_seq_len + 1.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            encoders.Linear(in_dim, out_dim),
        ),
        emsize=emsize,
        nhead=(emsize // 128),
        warmup_epochs=(epochs // 4),
        y_encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        batch_size=batch_size,
        scheduler=utils.get_cosine_schedule_with_warmup,
        extra_prior_kwargs_dict={
            # "num_workers": 10,
            "num_features": num_features,
            "hyperparameters": {
                **hps,
            },
        },
        epochs=epochs,
        lr=lr,
        bptt=seq_len,
        single_eval_pos_gen=single_eval_pos_gen,
        aggregate_k_gradients=1,
        nhid=(emsize * 2),
        steps_per_epoch=100,
        train_mixed_precision=False,
        saving_period=saving_period,
        checkpoint_file="lcpfn/trained_models/" + saving_name,
        outputs=outputs,
        num_outputs=num_outputs,
        decoder=partial(decoder, num_outputs=num_outputs),
        val_dataloader=val_dataloader,
        validation_period=validation_period,
        val_num_outputs=val_num_outputs,
        val_max_len_few_samples=5,
        val_min_len_many_samples=10,
        val_short_time=5,
        val_long_time=5,
        max_seq_len=max_seq_len,
        weighted_loss=weighted_loss,
    )

    return train.train(**config)
