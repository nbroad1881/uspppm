import gc
import math
import logging
import argparse
import datetime

import wandb
import torch
import datasets
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    get_scheduler,
    TrainingArguments,
    DataCollatorWithPadding,
)

from utils import (
    set_wandb_env_vars,
    reinit_model_weights,
    log_training_dynamics,
    compute_metrics,
    get_configs,
)
from data import DataModule
from modeling import get_pretrained


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on USPPPM dataset")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Config file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    config_file = args.config_file

    output = config_file.split(".")[0]
    cfg, args = get_configs(config_file)
    set_seed(args["seed"])
    set_wandb_env_vars(cfg)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
    mixed_precision = "fp16" if args["fp16"] else "bf16" if args["bf16"] else "fp32"
    accelerator = Accelerator(
        log_with=args["report_to"], logging_dir=args["output_dir"], mixed_precision=mixed_precision,
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    datamodule = DataModule(cfg)
    with accelerator.main_process_first():
        datamodule.prepare_datasets(add_idx=True)

    for fold in range(cfg["k_folds"]):

        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        train_dataset = datamodule.get_train_dataset(fold=fold)
        eval_dataset = datamodule.get_eval_dataset(fold=fold)

        train_dataset = train_dataset.map(lambda x: {"length": len(x["input_ids"])})
        max_len = 1

        eval_dataset = eval_dataset.map(lambda x: {"length": len(x["input_ids"])})
        max_eval_len = 1

        ds_cols = train_dataset.column_names
        keep_cols = {"input_ids", "attention_mask", "label", "idx"}
        

        data_collator = DataCollatorWithPadding(
            tokenizer=datamodule.tokenizer,
            padding=True,
            pad_to_multiple_of=8
        )

        train_dataloader = DataLoader(
            train_dataset.remove_columns([x for x in ds_cols if x not in keep_cols]),
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
        )
        eval_dataloader = DataLoader(
            eval_dataset.remove_columns([x for x in ds_cols if x not in keep_cols]),
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )

        model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"])
        model_config.update(
            {
                "num_labels": 1,
                "output_dropout_prob": cfg["dropout"],
                "num_concat": cfg["num_concat"],
                "meanmax_pooling": cfg["meanmax_pooling"],
                "mean_pooling": cfg["mean_pooling"],
                "max_pooling": cfg["max_pooling"],
                "attention_head": cfg["attention_head"],
                "multisample_dropout": cfg["multisample_dropout"],
                # "layer_norm_eps": cfg["layer_norm_eps"],
                "run_start": str(datetime.datetime.utcnow()),
                "output_hidden_states": True,
            }
        )

        model = get_pretrained(model_config, cfg["model_name_or_path"])

        reinit_model_weights(model, cfg["reinit_layers"], model_config)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_steps is None or args.max_steps == -1:
            args.max_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(
                args.max_steps / num_update_steps_per_epoch
            )

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_steps,
        )

        # Prepare everything with our `accelerator`.
        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to initialize the trackers we use, and also store our configuration
        if "wandb" in args.report_to:
            experiment_config = {**cfg, **args.to_dict()}
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config[
                "lr_scheduler_type"
            ]
            accelerator.init_trackers(args.output_dir, experiment_config)

        # Train!
        total_batch_size = (
            args.per_device_train_batch_size
            * accelerator.num_processes
            * args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {args.max_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(args.max_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        to_save = {}

        for epoch in range(args.num_train_epochs):
            model.train()
            if "wandb" in args.report_to:
                total_loss = 0

            train_probas = []
            train_labels = []
            train_ids = []

            for step, batch in enumerate(train_dataloader):

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                temp_logits = outputs["probas"].detach().cpu().numpy().ravel().tolist()
                temp_labels = batch["labels"].detach().cpu().numpy().ravel().tolist()
                temp_ids = (
                    np.tile(
                        batch["idx"].detach().cpu().numpy().reshape((-1, 1)),
                        (1, outputs.logits.shape[1]),
                    )
                    .ravel()
                    .tolist()
                )
                

                train_probas.extend(temp_logits)
                train_labels.extend(temp_labels)
                train_ids.extend(temp_ids)

                loss = outputs.loss
                # We keep track of the loss at each epoch
                if "wandb" in args.report_to:
                    total_loss += loss.detach().float()

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= args.max_steps:
                    break

            log_training_dynamics(
                output_dir=args.output_dir,
                epoch=epoch,
                train_ids=train_ids,
                train_probas=train_probas,
                train_golds=train_labels,
            )

            del train_probas, train_labels, train_ids
            gc.collect()

            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    bs = batch["input_ids"].size(0)
                    seq_len = batch["input_ids"].size(-1)

                    temp = (
                        outputs["probas"].detach().cpu().numpy().squeeze().tolist()
                    )
                    predictions.extend(temp)
                    labels.extend(batch["labels"].detach().cpu().squeeze().tolist())

            # predictions = np.vstack(predictions)
            predictions = np.array(predictions)

            eval_metrics = compute_metrics(((None, predictions), labels))
            
            del predictions, labels
            gc.collect()

            if "wandb" in args.report_to:
                accelerator.log(
                    {
                        "train_loss": total_loss,
                        "epoch": epoch,
                        "step": completed_steps,
                        **{f"eval/{k}": v for k, v in eval_metrics.items()},
                    },
                )
                
            torch.cuda.empty_cache()

        if "wandb" in args.report_to:
            wandb.finish()
            
        del model
        gc.collect()
        torch.cuda.empty_cache()