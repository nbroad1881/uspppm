import os
import re
import yaml
from pathlib import Path
from typing import List, Any, Optional, Tuple

import torch
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from transformers import (
    get_scheduler,
    PreTrainedTokenizerBase
)
from transformers.utils import logging
import bitsandbytes as bnb

logger = logging.get_logger(__name__)

def fix_e(cfg):

    def fix(value):
        pattern = r"\d+e\-\d+"
        if re.search(pattern, value):
            return eval(value)
        return value


    for k, v in cfg.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, str):
                    cfg[k][kk] = fix(vv)
        elif isinstance(v, str):
            cfg[k] = fix(v)
    
    return cfg
    
    

def remove_defaults(cfg):
    to_remove = []
    args = cfg["training_arguments"]
    for key, value in args.items():
        if value == "<default>":
            to_remove.append(key)
    
    for key in to_remove:
        del args[key]

def get_configs(filename, filepath="./configs"):

    file = Path(filepath) / filename
    with open(file) as fp:
        cfg = yaml.safe_load(fp)

    
    remove_defaults(cfg)
    cfg = fix_e(cfg)

    # cfg["training_arguments"]["dataloader_num_workers"] = cfg["num_proc"]

    training_args = cfg.pop("training_arguments")
    return cfg, training_args


def set_wandb_env_vars(cfg):
    os.environ["WANDB_ENTITY"] = cfg.get("entity", "")
    os.environ["WANDB_PROJECT"] = cfg.get("project", "")
    os.environ["WANDB_RUN_GROUP"] = cfg.get("group", "")
    os.environ["WANDB_JOB_TYPE"] = cfg.get("job_type", "")
    os.environ["WANDB_NOTES"] = cfg.get("notes", "")
    os.environ["WANDB_TAGS"] = ",".join(cfg.get("tags", ""))



def compute_metrics(eval_preds):

    (_, probas), labels = eval_preds
    corr, _ = pearsonr(probas.squeeze(), labels) 
    
    return {
        'proba_mse': mean_squared_error(labels, probas.squeeze()), 'proba_pearson': corr,
    }
    
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
    train_probas: List[List[float]],
    train_golds: List[int],
):
    """
    For dataset cartography
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """

    td_df = pd.DataFrame(
        {"guid": train_ids, f"logits_epoch_{epoch}": train_probas, "gold": train_golds}
    )

    logging_dir = os.path.join(output_dir, f"training_dynamics")
    # Create directory for logging training dynamics, if it doesn't already exist.
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Training Dynamics logged to {epoch_file_name}")


@dataclass
class OnlyMaskingCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    <Tip>
    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.
    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels