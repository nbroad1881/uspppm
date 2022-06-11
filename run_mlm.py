import argparse
from itertools import chain
import datetime

from datasets import load_metric, load_dataset

import wandb
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback


from utils import (
    get_configs,
    set_wandb_env_vars,
    freeze_layers,
    OnlyMaskingCollator,
    create_optimizer,
    create_scheduler,
)
from callbacks import NewWandbCB
from deberta import DebertaForMaskedLM, DebertaV2ForMaskedLM

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

    dataset = load_dataset("big_patent", "all")

    if cfg["DEBUG"]:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(1000))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= cfg["max_seq_length"]:
            total_length = (total_length // cfg["max_seq_length"]) * cfg[
                "max_seq_length"
            ]
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + cfg["max_seq_length"]]
                for i in range(0, total_length, cfg["max_seq_length"])
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    args["output_dir"] = output

    args = TrainingArguments(**args)

    with args.main_process_first(desc="dataset pre-processing"):
        dataset = dataset.map(
            lambda x: tokenizer(x["abstract"], return_special_tokens_mask=True),
            batched=True,
            num_proc=cfg["num_proc"],
            remove_columns=dataset["train"].column_names,
        )

        dataset = dataset.map(
            group_texts,
            batched=True,
            num_proc=cfg["num_proc"],
            remove_columns=dataset["train"].column_names,
        )

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    wb_callback = NewWandbCB(cfg)
    callbacks = [wb_callback]

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")

    model_config = AutoConfig.from_pretrained(cfg["model_name_or_path"])
    model_config.update(
        {
            "output_dropout": cfg["dropout"],
            # "layer_norm_eps": cfg["layer_norm_eps"],
            "run_start": str(datetime.datetime.utcnow()),
        }
    )

    if "deberta" in cfg["model_name_or_path"]:
        if "v2" in cfg["model_name_or_path"] or "v3" in cfg["model_name_or_path"]:
            model_fn = DebertaV2ForMaskedLM
        else:
            model_fn = DebertaForMaskedLM
    else:
        model_fn = AutoModelForMaskedLM

    model = model_fn.from_pretrained(cfg["model_name_or_path"], config=model_config)

    model.resize_token_embeddings(len(tokenizer))

    freeze_layers(
        getattr(model, model.config.model_type.split("-")[0]),
        cfg["n_frozen_layers"],
        cfg.get("freeze_embeds", True),
    )

    data_collator = OnlyMaskingCollator(
        tokenizer=tokenizer,
        return_tensors="pt",
        mlm_probability=cfg["masking_prob"],
    )

    steps_per_epoch = (
        len(train_dataset)
        // args.per_device_train_batch_size
        // cfg["n_gpu"]
        // args.gradient_accumulation_steps
    )
    num_training_steps = steps_per_epoch * args.num_train_epochs

    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(num_training_steps, optimizer, args)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    trainer.remove_callback(WandbCallback)

    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    model.config.update({"wandb_id": wandb.run.id, "wandb_name": wandb.run.name})
    model.config.save_pretrained(args.output_dir)

    wandb.finish()
