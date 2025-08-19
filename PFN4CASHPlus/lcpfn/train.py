import itertools
import time
from contextlib import nullcontext

import torch
from torch import nn

from lcpfn import utils
from lcpfn.transformer import TransformerModel
from lcpfn.bar_distribution import (
    BarDistribution,
)
from lcpfn.utils import (
    get_cosine_schedule_with_warmup,
    get_openai_lr,
)
from lcpfn import positional_encodings
#from lcpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler

import wandb
from bandit.bandit_validation import validate_model

class Losses:
    gaussian = nn.GaussianNLLLoss(full=True, reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = lambda num_classes: nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(num_classes)
    )
    bce = nn.BCEWithLogitsLoss(reduction="none")
    get_BarDistribution = BarDistribution


def train(
    priordataloader_class,
    criterion,
    encoder_generator,
    emsize=200,
    nhid=200,
    nlayers=6,
    nhead=2,
    dropout=0.2,
    epochs=10,
    steps_per_epoch=100,
    batch_size=200,
    bptt=10,
    lr=None,
    weight_decay=0.0,
    warmup_epochs=10,
    input_normalization=False,
    y_encoder_generator=None,
    pos_encoder_generator=None,
    decoder=None,
    extra_prior_kwargs_dict={},
    scheduler=get_cosine_schedule_with_warmup,
    load_weights_from_this_state_dict=None,
    single_eval_pos_gen=None,
    bptt_extra_samples=None,
    gpu_device="cuda:0",
    aggregate_k_gradients=1,
    verbose=True,
    style_encoder_generator=None,
    epoch_callback=None,
    initializer=None,
    train_mixed_precision=False,
    saving_period=10,
    checkpoint_file=None,
    output_path=None,
    outputs = ['max'],
    num_outputs=1,
    val_dataloader=None,
    validation_period=1,
    val_num_outputs=1,
    val_max_len_few_samples=5,
    val_min_len_many_samples=20,
    val_short_time=5,
    val_long_time=5,
    max_seq_len = 200,
    weighted_loss = False,
    **model_extra_args,
):
    device = gpu_device if torch.cuda.is_available() else "cpu:0"
    print(f"Using {device} device")
    single_eval_pos_gen = (
        single_eval_pos_gen
        if callable(single_eval_pos_gen)
        else lambda: single_eval_pos_gen
    )

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt

    dl = priordataloader_class(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=bptt + (bptt_extra_samples if bptt_extra_samples else 0),
        device=device,
        **extra_prior_kwargs_dict,
    )

    if val_dataloader != None :
        few_samples_single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(
            val_max_len_few_samples, min_len=1
        )
        many_samples_single_eval_pos_gen = utils.get_uniform_single_eval_pos_sampler(
            bptt, min_len=val_max_len_few_samples
        )

        def few_samples_eval_pos_seq_len_sampler():
            return few_samples_single_eval_pos_gen(), bptt

        def many_samples_eval_pos_seq_len_sampler():
            return many_samples_single_eval_pos_gen(), bptt

        few_samples_validation_dl = val_dataloader(
            num_steps=steps_per_epoch,
            batch_size=batch_size,
            eval_pos_seq_len_sampler=few_samples_eval_pos_seq_len_sampler,
            seq_len_maximum=bptt,
            device=device,
            num_features=1,
        )
        many_samples_validation_dl = val_dataloader(
            num_steps=steps_per_epoch,
            batch_size=batch_size,
            eval_pos_seq_len_sampler=many_samples_eval_pos_seq_len_sampler,
            seq_len_maximum=bptt,
            device=device,
            num_features=1,
        )

    encoder = encoder_generator(dl.num_features, emsize)
    style_def = next(iter(dl))[0][
        0
    ]  # This is (style, x, y), target with x and y with batch size
    print(f"Style definition: {style_def}")
    style_encoder = (
        style_encoder_generator(hyperparameter_definitions=style_def[0], em_size=emsize)
        if (style_def is not None)
        else None
    )
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif (
        isinstance(criterion, BarDistribution)
        or "BarDistribution" in criterion.__class__.__name__
    ):  # TODO remove this fix (only for dev)
        n_out = criterion.num_bars
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    model = TransformerModel(
        encoder,
        n_out,
        emsize,
        nhead,
        nhid,
        nlayers,
        dropout,
        style_encoder=style_encoder,
        y_encoder=y_encoder_generator(num_outputs, emsize),
        input_normalization=input_normalization,
        pos_encoder=(
            pos_encoder_generator or positional_encodings.NoPositionalEncoding
        )(emsize, bptt * 2),
        decoder=decoder,
        init_method=initializer,
        **model_extra_args,
    )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)

    print(
        f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters"
    )

    #print("model:", model)
    model.to(device)
    # learning rate
    if lr is None:
        lr = get_openai_lr(model)
        print(f"Using OpenAI max lr of {lr}.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(
        optimizer, warmup_epochs, epochs if epochs is not None else 100
    )  # when training for fixed time lr schedule takes 100 steps
    
    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0
        total_positional_losses = 0
        total_positional_losses_recorded = 0.0
        loss_weighting = torch.ones(bptt, 1).to(device)

        before_get_batch = time.time()
        assert (
            len(dl) % aggregate_k_gradients == 0
        ), "Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it."
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            
            cm = nullcontext()
            with cm:
                time_to_get_batch = time.time() - before_get_batch
                before_forward = time.time()

                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = model(
                        tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                        if isinstance(data, tuple)
                        else data.to(device),
                        single_eval_pos=single_eval_pos,
                    )
                    
                    forward_time = time.time() - before_forward

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]
                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert (
                            output.shape[-1] == 2
                        ), "need to write a little bit of code to handle multiple regression targets at once"

                        mean_pred = output[..., 0]
                        var_pred = output[..., 1].abs()
                        losses = criterion(
                            mean_pred.flatten(),
                            targets.to(device).flatten(),
                            var=var_pred.flatten(),
                        )
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        losses = criterion(
                            output.flatten(), targets.to(device).flatten()
                        )
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        losses = criterion(
                            output.reshape(-1, n_out),
                            targets.to(device).long().flatten(),
                        )
                    else:
                        losses = criterion(output, targets)
                    losses = losses.view(*output.shape[0:2])
                    if(weighted_loss):
                        weights = loss_weighting[single_eval_pos:]/loss_weighting[single_eval_pos:].sum()
                        loss = (weights * losses).mean() / aggregate_k_gradients
                        loss_weighting[single_eval_pos:] = 0.9 * loss_weighting[single_eval_pos:]
                    else:
                        loss = losses.mean() / aggregate_k_gradients
                if scaler:
                    loss = scaler.scale(loss)

                #print((output.isnan().any()), data[2].isnan().any(), loss)
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    try:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach()
                    total_positional_losses += (
                        losses.mean(1).cpu().detach()
                        if single_eval_pos is None
                        else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                        * losses[: bptt - single_eval_pos].mean().cpu().detach()
                    )

                    total_positional_losses_recorded += (
                        torch.ones(bptt)
                        if single_eval_pos is None
                        else nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                    )

            before_get_batch = time.time()
        return (
            total_loss / steps_per_epoch,
            (total_positional_losses / total_positional_losses_recorded).tolist(),
            time_to_get_batch,
            forward_time,
            step_time,
        )

    total_loss = float("inf")
    total_positional_losses = float("inf")
    list_losses = []


    def validation(validation_dl):
        results = {}
        val_loss = 0.0
        short_time_val_loss = 0.0
        long_time_val_loss = 0.0

        val_square_error = 0.0
        short_time_val_square_error = 0.0
        long_time_val_square_error = 0.0

        task_lcb_error = 0.0
        short_time_task_lcb_error = 0.0
        long_time_task_lcb_error = 0.0

        task_ucb_error = 0.0
        short_time_task_ucb_error = 0.0
        long_time_task_ucb_error = 0.0


        for batch, (data, targets, single_eval_pos) in enumerate(validation_dl):
            with torch.no_grad():
                output = model(
                    tuple(e.to(device) if torch.is_tensor(e) else e for e in data)
                    if isinstance(data, tuple)
                    else data.to(device),
                    single_eval_pos=single_eval_pos,
                )
                if single_eval_pos is not None:
                    lcb_target = targets[single_eval_pos - 1]
                    targets = targets[single_eval_pos:]

            losses = model.criterion(output[..., 0:val_num_outputs], targets[..., 0:val_num_outputs])
            losses = losses.view(*output.shape[0:2]).cpu().detach()
            short_time_val_loss += losses[: min(val_short_time, len(losses))].mean()
            long_time_val_loss += losses[
                max(0, len(losses) - val_long_time) :
            ].mean()
            val_loss += losses.mean()

            square_error  = ((model.criterion.mean(output[..., 0]) -  targets[..., 0])**2).cpu().detach()

            lcb = model.criterion.icdf(output[..., 0], left_prob=0.01)
            task_lcb_diff = ((lcb < lcb_target[...,0]).float() * (lcb - lcb_target[...,0]) ** 2).cpu().detach()
            #print(lcb.shape, lcb_target[..., 0].shape)

            ucb = model.criterion.icdf(output[..., 0], left_prob=0.99)
            #print(ucb.shape, targets[-1, :, 0].shape)
            task_ucb_diff = ((ucb > targets[-1, :, 0]) * (ucb - targets[-1, :, 0]) ** 2 ).cpu().detach()

            val_square_error += torch.mean(square_error) #/batch_size
            short_time_val_square_error += torch.mean(square_error[: min(val_short_time, len(losses))])#/val_short_time
            long_time_val_square_error += torch.mean(square_error[max(0, len(losses) - val_long_time) :])#/val_long_time

            task_lcb_error += torch.mean(task_lcb_diff) #/batch_size
            short_time_task_lcb_error += torch.mean(task_lcb_diff[: min(val_short_time, len(losses))])#/val_short_time
            long_time_task_lcb_error += torch.mean(task_lcb_diff[max(0, len(losses) - val_long_time) :])#/val_long_time

            task_ucb_error += torch.mean(task_ucb_diff) #/batch_size
            short_time_task_ucb_error += torch.mean(task_ucb_diff[: min(val_short_time, len(losses))])#/val_short_time
            long_time_task_ucb_error += torch.mean(task_ucb_diff[max(0, len(losses) - val_long_time) :])#/val_long_time

        results["val_loss"] = val_loss / steps_per_epoch 
        results["short_time_val_loss"] = short_time_val_loss / steps_per_epoch
        results["long_time_val_loss"] = long_time_val_loss / steps_per_epoch

        results["val_RMSE"] = torch.sqrt(val_square_error / steps_per_epoch)
        results["short_time_val_RMSE"] = torch.sqrt(
            short_time_val_square_error / steps_per_epoch
        )
        results["long_time_val_RMSE"] = torch.sqrt(
            long_time_val_square_error / steps_per_epoch
        )

        results["task_lcb_error"] = task_lcb_error / steps_per_epoch
        results["short_time_task_lcb_error"] = (
            short_time_task_lcb_error / steps_per_epoch
        )
        results["long_time_task_lcb_error"] = long_time_task_lcb_error / steps_per_epoch

        results["task_ucb_error"] = task_ucb_error / steps_per_epoch
        results["short_time_task_ucb_error"] = (
            short_time_task_ucb_error / steps_per_epoch
        )
        results["long_time_task_ucb_error"] = long_time_task_ucb_error / steps_per_epoch

        return results

    try:
        for epoch in range(1, epochs + 1) if epochs is not None else itertools.count(1):
            epoch_start_time = time.time()
            (
                total_loss,
                total_positional_losses,
                time_to_get_batch,
                forward_time,
                step_time,
            ) = train_epoch()
            list_losses.append(total_loss.item())

            metrics = {}
            if val_dataloader != None and epoch % validation_period == 0:
                few_samples_validation_start = time.time()
                results = validation(few_samples_validation_dl)
                few_samples_validation_time = time.time() - few_samples_validation_start
                metrics.update({f"few_samples_{k}": v for k, v in results.items()})
                metrics["few_samples_validation_time"] = few_samples_validation_time

                many_samples_validation_start= time.time()
                results = validation(many_samples_validation_dl)
                many_samples_validation_time = time.time() - many_samples_validation_start
                metrics.update({f"many_samples_{k}": v for k, v in results.items()})
                metrics["many_samples_validation_time"] = many_samples_validation_time

            if epoch % saving_period == 0 or epoch == 1:
                validate_datasets = {
                    "YaHPOGym(NS)": "non_stationary_YaHPOGym_distilled",
                    "TabRepoRaw(Cl)": "classification_TabRepoRaw_distilled",
                }
                for key, item in validate_datasets.items():
                    validate_model(model, device, item, key, outputs)


            if epoch % saving_period == 0 and checkpoint_file is not None:

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                }
                torch.save(checkpoint, checkpoint_file)
                full_model_path = checkpoint_file.split(".")[0] + "_full_model.pt"
                torch.save(model, full_model_path)


            if verbose:
                print("-" * 89)
                res = (f"| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | "
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f" data time {time_to_get_batch:5.2f} step time {step_time:5.2f}"
                    f" forward time {forward_time:5.2f}")
                res += "| ".join(f"{k}: {v}" for k, v in metrics.items())
                print(res)
                print("-" * 89)
            if wandb:
                metrics.update({"epoch":epoch, "time":(time.time() - epoch_start_time),"mean_loss":total_loss, "pos_losses":total_positional_losses,
                "learning_rate": scheduler.get_last_lr()[0], "data_time":time_to_get_batch, "step_time":step_time, "forward_time":forward_time})
                wandb.log(metrics)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None:
                epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            dl = None
        if output_path is not None:
            torch.save(model.to("cpu"), output_path)
            print("Checkpoint stored at ", output_path)
        return total_loss, total_positional_losses, model.to("cpu"), dl
