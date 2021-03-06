import os
import datetime
import argparse
from functools import partial

import wandb
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    DataCollatorWithPadding,
    AutoModelForSeq2SeqLM,
)
from transformers.trainer_utils import set_seed
from transformers.integrations import WandbCallback


from callbacks import NewWandbCB, SaveCallback, BasicSWACallback
from utils import (
    get_configs,
    set_wandb_env_vars,
    reinit_model_weights,
    seq2seq_compute_metrics,
    create_optimizer,
    create_scheduler,
    push_to_hub,
)
from data import DataModule
from modeling import (
    Seq2SeqUSPPPMModel,
)


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

    dm = DataModule(cfg)

    dm.prepare_datasets()

    for fold in range(1, cfg["k_folds"]):

        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        metric_to_track = "eval_proba_pearson"
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"],
            metric_name=metric_to_track,
            weights_only=True,
        )

        callbacks = [wb_callback, save_callback]
        if cfg["use_swa"]:
            callbacks.append(
                BasicSWACallback(
                    start_after=cfg["swa_start_after"], save_every=cfg["swa_save_every"]
                )
            )

        train_dataset = dm.get_train_dataset(fold)
        eval_dataset = dm.get_eval_dataset(fold)
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Eval dataset length: {len(eval_dataset)}")

        print(
            "Decode inputs from train_dataset",
            dm.tokenizer.decode(train_dataset[0]["input_ids"]),
        )

        if "cocolm" in cfg["model_name_or_path"]:
            cfg_class = COCOLMConfig
        else:
            cfg_class = AutoConfig

        model_config = cfg_class.from_pretrained(
            cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )
        model_config.update(
            {
                "run_start": str(datetime.datetime.utcnow()),
            }
        )
        
        # ???maximum 2411
        # ???very 182
        # ???some 128
        # ???little 385
        # ???no 150
        mapping = {
                2411: 1.0,
                182: 0.75,
                128: 0.5,
                385: 0.25,
                150: 0.0,
            }
        
        compute_metrics = partial(seq2seq_compute_metrics, mapping=mapping)

        model = Seq2SeqUSPPPMModel(model_config, id2score=mapping)
        model.backbone = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name_or_path"])

        optimizer = create_optimizer(model, args)

        steps_per_epoch = (
            len(train_dataset)
            // args.per_device_train_batch_size
            // cfg["n_gpu"]
            // args.gradient_accumulation_steps
        )
        num_training_steps = steps_per_epoch * args.num_train_epochs

        scheduler = create_scheduler(num_training_steps, optimizer, args)

        collator = DataCollatorWithPadding(tokenizer=dm.tokenizer, pad_to_multiple_of=8)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=dm.tokenizer,
            callbacks=callbacks,
            data_collator=collator,
            optimizers=(optimizer, scheduler),
        )

        trainer.remove_callback(WandbCallback)

        trainer.train()

        if cfg.get("use_swa"):
            trainer.model.load_state_dict(
                torch.load(os.path.join(args.output_dir, "swa_weights.bin"))
            )
            eval_results = trainer.evaluate()

        trainer.log(
            {
                f"best_{metric_to_track}": trainer.model.config.to_dict().get(
                    f"best_{metric_to_track}"
                )
            }
        )
        model.config.update({"wandb_id": wandb.run.id, "wandb_name": wandb.run.name})
        model.config.save_pretrained(args.output_dir)

        if args.push_to_hub:
            push_to_hub(trainer)

        wandb.finish()

        torch.cuda.empty_cache()
