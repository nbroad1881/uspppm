import os
from typing import List

import torch
import pandas as pd

from transformers import (
    get_scheduler,
)
from transformers.utils import logging
import bitsandbytes as bnb

logger = logging.get_logger(__name__)


def set_wandb_env_vars(cfg):
    os.environ["WANDB_ENTITY"] = cfg.get("entity", "")
    os.environ["WANDB_PROJECT"] = cfg.get("project", "")
    os.environ["WANDB_RUN_GROUP"] = cfg.get("group", "")
    os.environ["WANDB_JOB_TYPE"] = cfg.get("job_type", "")
    os.environ["WANDB_NOTES"] = cfg.get("notes", "")
    os.environ["WANDB_TAGS"] = ",".join(cfg.get("tags", ""))


def reinit_model_weights(model, n_layers, config):

    backbone = model.backbone
    if config.model_type == "bart":
        std = config.init_std
    else:
        std = config.initializer_range

    if n_layers > 0:
        if config.model_type == "bart":
            encoder_layers = backbone.encoder.layers
            decoder_layers = backbone.decoder.layers

            reinit_layers(encoder_layers, n_layers, std)
            reinit_layers(decoder_layers, n_layers, std)
        else:
            encoder_layers = backbone.encoder.layer
            reinit_layers(encoder_layers, n_layers, std)

    reinit_modules([model.output], std)


def reinit_layers(layers, n_layers, std):
    for layer in layers[-n_layers:]:
        reinit_modules(layer.modules(), std)


def reinit_modules(modules, std, reinit_embeddings=False):
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif reinit_embeddings and isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def layerwise_learning_rate(model, lr=3e-5, wd=0.01, alpha=0.8):
    model_type = model.backbone_name

    layers = (
        [getattr(model, model_type).embeddings]
        + [getattr(model, model_type).encoder.layer]
        + [model.output]
    )
    layers.reverse()

    optimizer_grouped_parameters = []

    for i, layer in enumerate(layers):
        # This keeps top layer = lr
        if i > 0:
            lr *= alpha
        optimizer_grouped_parameters += uniform_learning_rate(layer, wd)

    return optimizer_grouped_parameters


def create_optimizer(model, train_args):
    return bnb.optim.Adam8bit(
        uniform_learning_rate(model, train_args.learning_rate, train_args.weight_decay),
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_epsilon,
    )


def create_scheduler(num_training_steps, optimizer, train_args, **kwargs):

    # if self.run_config.lr_scheduler == "step":
    #     milestones = [m * num_training_steps for m in self.run_config.lr_milestones]
    #     scheduler = lr_scheduler.MultiStepLR(
    #         optimizer,
    #         milestones=milestones,
    #         gamma=self.run_config.lr_gamma,
    #     )

    # else:
    if train_args.warmup_ratio > 0:
        warmup_steps = num_training_steps * train_args.warmup_ratio
    else:
        warmup_steps = train_args.warmup_steps

    scheduler = get_scheduler(
        train_args.lr_scheduler_type,
        optimizer,
        warmup_steps,
        num_training_steps,
    )

    # if self.run_config.use_swa:
    #     self.swa_scheduler = SWALR(
    #         optimizer,
    #         swa_lr=self.run_config.swa_lr,
    #         anneal_epochs=self.run_config.swa_anneal_steps,
    #     )
    return scheduler


def uniform_learning_rate(model, lr, wd=0.01):

    no_decay = ["bias", "LayerNorm.weight"]
    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]


def freeze_layers(model, n_layers, freeze_embeds=True):
    if freeze_embeds:
        model.embeddings.requires_grad_(False)

    model.encoder.layer[:n_layers].requires_grad_(False)


def log_training_dynamics(
    output_dir: os.path,
    epoch: int,
    train_ids: List[int],
    train_logits: List[List[float]],
    train_golds: List[int],
):
    """
    For dataset cartography
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """

    td_df = pd.DataFrame(
        {"guid": train_ids, f"logits_epoch_{epoch}": train_logits, "gold": train_golds}
    )

    logging_dir = os.path.join(output_dir, f"training_dynamics")
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Training Dynamics logged to {epoch_file_name}")
