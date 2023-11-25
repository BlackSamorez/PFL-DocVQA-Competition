import copy
import json
import os
import random
from collections import OrderedDict
from communication.log_communication import log_communication
from communication.compute_tensor_size import get_bytes_for_tensor

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn

from tqdm import tqdm
from build_utils import (build_dataset, build_model, build_optimizer, build_provider_dataset)
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape)
from eval import evaluate  # fl_centralized_evaluation
from logger import Logger
from metrics import Evaluator
from checkpoint import save_model
from utils import load_config, parse_args, seed_everything
from utils_parallel import get_parameters_from_model, set_parameters_model, weighted_average
from collections import OrderedDict


def fl_train(data_loaders, model, optimizer, evaluator, logger, client_id, fl_config):
    """
    Trains and returns the updated weights.
    """
    model.model.train()
    param_keys = list(model.model.state_dict().keys())
    parameters = copy.deepcopy(list(model.model.state_dict().values()))

    keyed_parameters = {n: p.requires_grad for n, p in model.model.named_parameters()}
    frozen_parameters = [not keyed_parameters[n] if n in keyed_parameters else False for n, p in model.model.state_dict().items()]

    logger.current_epoch += 1
    logger.bits_downlink += sum([get_bytes_for_tensor(w_0) for w_0, is_frozen in zip(parameters, frozen_parameters) if not is_frozen])

    agg_update = None
    if not config.use_dp and len(data_loaders) > 1:
        raise ValueError("Non private training should only use one data loader.")

    total_training_steps = sum([len(data_loader) for data_loader in data_loaders]) * config.fl_params.iterations_per_fl_round
    total_training_samples = sum([len(data_loader.dataset) for data_loader in data_loaders]) * config.fl_params.iterations_per_fl_round
    pbar = tqdm(total=total_training_steps)

    total_loss = 0
    fl_round_acc = 0
    fl_round_anls = 0

    for provider_dataloader in data_loaders:
        # Set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        model.model.load_state_dict(state_dict, strict=True)
        model.model.train()

        # Reset the optimizer
        if config.use_dp:
            optimizer = build_optimizer(model, config)
        scaler = torch.cuda.amp.GradScaler()

        # Perform N provider iterations (each provider has their own dataloader in the non-private case)
        for iter in range(config.fl_params.iterations_per_fl_round):
            for batch_idx, batch in enumerate(provider_dataloader):

                gt_answers = batch['answers']
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs, pred_answers, answer_conf = model.forward(batch, return_pred_answer=True)
                loss = outputs.loss

                # total_loss += loss.item() / len(batch['question_id'])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # lr_scheduler.step()
                optimizer.zero_grad()

                metric = evaluator.get_metrics(gt_answers, pred_answers)

                total_loss += outputs.loss.item()
                fl_round_acc += np.sum(metric['accuracy'])
                fl_round_anls += np.sum(metric['anls'])

                log_dict = {
                    'Train/Batch loss': outputs.loss.item(),
                    'Train/Batch Accuracy': np.mean(metric['accuracy']),
                    'Train/Batch ANLS': np.mean(metric['anls']),
                    'lr': optimizer.param_groups[0]['lr'],
                    'grad_scale': scaler.get_scale(),
                }
                log_dict.update({f"{k}@{client_id}": v for k, v in log_dict.items()})

                logger.logger.log(log_dict)
                pbar.update()

        # After all the iterations:
        # Get the update
        new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]  # Get model update

        if config.use_dp:
            # flatten update
            shapes = get_shape(new_update)
            new_update = flatten_params(new_update)

            # clip update:
            new_update = clip_parameters(new_update, clip_norm=config.dp_params.sensitivity)

            # Aggregate (Avg)
            if agg_update is None:
                agg_update = new_update
            else:
                agg_update += new_update

    # Handle DP after all updates are done
    if config.use_dp:
        # Add the noise
        agg_update = add_dp_noise(agg_update, noise_multiplier=config.dp_params.noise_multiplier, sensitivity=config.dp_params.sensitivity)

        # Divide the noisy aggregated update by the number of providers (Average update).
        agg_update = torch.div(agg_update, len(data_loaders))

        # Add the noisy update to the original model
        agg_update = reconstruct_shape(agg_update, shapes)

        # Restore original weights (without noise) from frozen layers.
        agg_update = [upd if not is_frozen else 0 for upd, params, is_frozen in zip(agg_update, parameters, frozen_parameters)]

        # all([torch.all(params == new_params).item() == is_frozen for params, new_params, is_frozen in zip(parameters, agg_update, frozen_parameters)])  Restoration Test

    else:
        agg_update = new_update

    # upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0 in zip(agg_update, copy.deepcopy(parameters))]  # Send all weights
    upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0, is_frozen in zip(agg_update, copy.deepcopy(parameters), frozen_parameters) if not is_frozen]  # Send weights of NON-Frozen layers.

    pbar.close()
    
    # if fl_config["log_path"] is not None:
    if config.flower:
        # log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store model's weights bytes.
        log_communication(federated_round=fl_config.current_round, sender=client_id, receiver=-1, data=upd_weights, log_location=logger.comms_log_file)  # Store only communicated weights (sent parameters).
    logger.bits_uplink += sum([get_bytes_for_tensor(w_0) for w_0 in upd_weights])

    fl_round_log_dict = {
        'Train/FL Round loss': total_loss / total_training_samples,
        'Train/FL Round Accuracy': fl_round_acc / total_training_samples,
        'Train/FL Round ANLS': fl_round_anls / total_training_samples,
        'fl_round': logger.current_epoch,
        'bits_total': logger.bits_total, 
    }

    logger.logger.log(fl_round_log_dict)

    # Send the weights to the server
    return upd_weights


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    model = build_model(config)
    optimizer = build_optimizer(model, config=config)
    
    # for name, p in model.model.named_parameters():
    #     print(p.requires_grad, name)
    # exit(0)
    
    params = get_parameters_from_model(model)
    train_dataset = build_dataset(config, 'train', client_id=0)
    dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config=config)
    
    fl_train([dataloader], model, optimizer, evaluator, logger, 0, config)    
