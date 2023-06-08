import copy
import json
import os
import random
import warnings
from collections import OrderedDict
import gc

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from build_utils import (build_dataset, build_model, build_optimizer,
                         build_provider_dataset)
from datasets.BaseDataset import collate_fn
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters,
                                           flatten_params, get_shape,
                                           reconstruct)
from eval import evaluate
from logger import Logger
from metrics import Evaluator
from utils import load_config, parse_args, seed_everything
from utils_parallel import get_parameters, set_parameters, weighted_average

TRAIN_PRIVATE = False


def fl_train(data_loaders, model, optimizer, lr_scheduler, evaluator, logger, fl_config):
    """
    Trains and returns the updated weights.
    """
    model.model.train()
    param_keys = list(model.model.state_dict().keys())
    parameters = copy.deepcopy(list(model.model.state_dict().values()))
    sensitivity = config.sensitivity
    noise_multiplier = 0
    TRAIN_PRIVATE = False
    provider_iterations = 1

    warnings.warn("\n\n" + str(config) + "\n\n")
    warnings.warn("\n\n" + str(fl_config) + "\n\n")
    agg_update = None

    if not TRAIN_PRIVATE and len(data_loaders) > 1:
        raise ValueError("Non private training should only use one data loader.")

    for provider_dataloader in data_loaders:
        total_loss = 0

        # set model weights to state of beginning of federated round
        state_dict = OrderedDict({k: v for k, v in zip(param_keys, parameters)})
        model.model.load_state_dict(state_dict, strict=True)
        model.model.train()

        # perform n provider iterations (each provider has their own dataloader)
        for iter in range(provider_iterations):
            for batch_idx, batch in enumerate(tqdm(provider_dataloader)):
                gt_answers = batch['answers']
                outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)
                loss = outputs.loss + outputs.ret_loss if hasattr(outputs, 'ret_loss') else outputs.loss

                total_loss += loss.item() / len(batch['question_id'])
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                metric = evaluator.get_metrics(gt_answers, pred_answers)

                batch_acc = np.mean(metric['accuracy'])
                batch_anls = np.mean(metric['anls'])

                log_dict = {
                    'Train/Batch loss': outputs.loss.item(),
                    'Train/Batch Accuracy': batch_acc,
                    'Train/Batch ANLS': batch_anls,
                    'lr': optimizer.param_groups[0]['lr']
                }

                if hasattr(outputs, 'ret_loss'):
                    log_dict['Train/Batch retrieval loss'] = outputs.ret_loss.item()

                if 'answer_page_idx' in batch and None not in batch['answer_page_idx']:
                    ret_metric = evaluator.get_retrieval_metric(batch.get('answer_page_idx', None), pred_answer_page)
                    batch_ret_prec = np.mean(ret_metric)
                    log_dict['Train/Batch Ret. Prec.'] = batch_ret_prec

                logger.logger.log(log_dict)

        # After all the iterations:
        # Get the update
        new_update = [w - w_0 for w, w_0 in zip(list(model.model.state_dict().values()), parameters)]  # Get model update

        if TRAIN_PRIVATE:
            # flatten update
            shapes = get_shape(new_update)
            new_update = flatten_params(new_update)

            # clip update:
            new_update = clip_parameters(new_update, clip_norm=sensitivity)

            # Aggregate (Avg)
            if agg_update is None:
                agg_update = new_update
            else:
                agg_update += new_update

    # handle DP after all updates are done
    if TRAIN_PRIVATE:
        # Add the noise
        agg_update = add_dp_noise(agg_update, noise_multiplier=noise_multiplier, sensitity=sensitivity)

        # Divide the noisy aggregated update by the number of providers (100)
        agg_update = torch.div(agg_update, len(data_loaders))

        # Add the noisy update to the original model
        agg_update = reconstruct(agg_update, shapes)
    else:
        agg_update = new_update

    upd_weights = [torch.add(agg_upd, w_0).cpu() for agg_upd, w_0 in zip(agg_update, copy.deepcopy(parameters))]

    # Send the weights to the server
    logger.logger.log(log_dict, step=logger.current_epoch * logger.len_dataset + batch_idx)

    return upd_weights


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, optimizer, lr_scheduler, evaluator, logger, config):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.logger = logger
        self.logger.log_model_parameters(self.model)
        self.config = config

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # debug: write parameters to file each round
        with open("parameters.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, np.array([sum([np.sum(parameters[i]) for i in range(len(parameters))])]))
            np.savetxt(f, np.array([sum([np.mean(parameters[i]) for i in range(len(parameters))])]))
            np.savetxt(f, np.array([sum([np.std(parameters[i]) for i in range(len(parameters))])]))
        updated_weigths = fl_train(self.trainloader, self.model, self.optimizer, self.lr_scheduler, self.evaluator, self.logger, config)

        # debug: write updated weights to file each round
        with open("fit_weights.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, updated_weigths[0].numpy()[:10, 0])
            f.write(b"\n")
            np.savetxt(f, np.array([sum([np.sum(updated_weigths[i].numpy()) for i in range(len(updated_weigths))])]))
            np.savetxt(f, np.array([sum([np.mean(updated_weigths[i].numpy()) for i in range(len(updated_weigths))])]))
            np.savetxt(f, np.array([sum([np.std(updated_weigths[i].numpy()) for i in range(len(updated_weigths))])]))
        return updated_weigths, 1, {}  # TODO 1 ==> Number of selected clients.

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        accuracy, anls, ret_prec, _, _ = evaluate(self.valloader, self.model, self.evaluator,
                                                  config)  # data_loader, model, evaluator, **kwargs
        is_updated = self.evaluator.update_global_metrics(accuracy, anls, 0)
        self.logger.log_val_metrics(accuracy, anls, ret_prec, update_best=is_updated)

        return float(0), len(self.valloader), {"accuracy": float(accuracy), "anls": anls}


def client_fn(node_id):
    """Create a Flower client representing a single organization."""
    # pick a subset of providers
    provider_to_doc = json.load(open(config.provider_docs, 'r'))
    provider_to_doc = provider_to_doc["node_" + node_id]
    providers = random.sample(list(provider_to_doc.keys()), k=config.providers_per_fl_round)  # 50

    # create a list of train data loaders with one dataloader per provider
    if TRAIN_PRIVATE:
        train_datasets = [build_provider_dataset(config, 'train', provider_to_doc, provider, node_id) for provider in providers]
    else:
        train_datasets = [build_dataset(config, 'train', node_id=node_id)]

    train_data_loaders = [DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,
                                     collate_fn=collate_fn) for train_dataset in train_datasets]

    # create validation data loader
    val_dataset = build_dataset(config, 'val')
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer, lr_scheduler = build_optimizer(model, length_train_loader=len(train_data_loaders), config=config)

    evaluator = Evaluator(case_sensitive=False)
    logger = Logger(config=config)
    return FlowerClient(model, train_data_loaders, val_data_loader, optimizer, lr_scheduler, evaluator, logger, config)


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "local_epochs": 2,
    }
    return config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    NUM_CLIENTS = config.num_clients
    model = build_model(config)  # TODO Should already be in CUDA
    params = get_parameters(model)

    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=config.client_sampling_probability,  # Sample 100% of available clients for training
        fraction_fit=0.33,  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 50% of available clients for evaluation
        min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
        min_evaluate_clients=NUM_CLIENTS,  # Never sample less than 5 clients for evaluation
        min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        on_fit_config_fn=fit_config
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if config.device == "cuda":
        client_resources = {"num_gpus": 1}  # TODO Check number of GPUs

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"local_mode": True}
    )
