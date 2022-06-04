import os
import datetime
import argparse

import wandb
import torch
from transformers import Trainer, TrainingArguments, AutoConfig
from transformers.trainer_utils import set_seed
from transformers.integrations import WandbCallback


from callbacks import NewWandbCB, SaveCallback, BasicSWACallback
from utils import get_configs, set_wandb_env_vars, reinit_model_weights, compute_metrics
from data import DataModule
from modeling import (
    get_pretrained,
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

    for fold in range(cfg["k_folds"]):

        cfg, args = get_configs(config_file)
        cfg["fold"] = fold
        args["output_dir"] = f"{output}-f{fold}"

        args = TrainingArguments(**args)

        # Callbacks
        wb_callback = NewWandbCB(cfg)
        metric_to_track = "eval_logit_pearson"
        save_callback = SaveCallback(
            min_score_to_save=cfg["min_score_to_save"],
            metric_name=metric_to_track,
            weights_only=False,
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

        model_config = AutoConfig.from_pretrained(
            cfg["model_name_or_path"],
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )
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

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=dm.tokenizer,
            callbacks=callbacks,
        )

        trainer.remove_callback(WandbCallback)

        trainer.train()

        if cfg.get("use_swa"):
            trainer.model.load_state_dict(
                torch.load(os.path.join(args.output_dir, "swa_weights.bin"))
            )
            eval_results = trainer.evaluate()

        trainer.log({f"best_{metric_to_track}": trainer.model.config.to_dict().get(f"best_{metric_to_track}")})
        model.config.update({"wandb_id": wandb.run.id, "wandb_name": wandb.run.name})
        model.config.save_pretrained(args.output_dir)

        if args.push_to_hub:
            trainer.push_to_hub()

        wandb.finish()

        torch.cuda.empty_cache()
