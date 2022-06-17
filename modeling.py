import os
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
import numpy as np
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
)
from transformers.utils import ModelOutput
from cocolm.modeling_cocolm import COCOLMModel


class USPPPMConfig(PretrainedConfig):
    model_type = "uspppm"

    def __init__(
        self,
        prompt: str = "natural",
        num_concat: int = 1,
        pooling: str = "cls",
        output_hidden_dim: int = 512,
        **kwargs,
    ):
        self.prompt = prompt
        self.num_concat = num_concat
        self.pooling = pooling
        self.output_hidden_dim = output_hidden_dim
        super().__init__(**kwargs)


class USPPPMModel(PreTrainedModel):

    config_class = USPPPMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if "COCO" in str(config.__class__):
            self.backbone = COCOLMModel(config)
        else:
            self.backbone = AutoModel.from_config(config)

        self.classification_head = [ConcatHiddenStates(config.num_concat)]
        input_hidden_size = config.hidden_size * config.num_concat

        if config.pooling == "meanmax":
            input_hidden_size *= 2
            self.classification_head.append(MeanMaxPoolHead())
        elif config.pooling == "mean":
            self.classification_head.append(MeanPoolHead())
        elif config.pooling == "max":
            self.classification_head.append(MaxPoolHead())
        elif config.pooling == "attn":
            self.classification_head.append(
                AttentionHead(
                    input_hidden_size=input_hidden_size,
                )
            )
        else:
            self.classification_head.append(CLSHead())

        self.classification_head = nn.ModuleList(self.classification_head)

        self.dropout = nn.Dropout(config.output_dropout_prob)
        if config.multisample_dropout:
            self.multisample_dropout = MultiSampleDropout(config.multisample_dropout, "regression", 1)

        if config.output_layer_norm:
            self.ln = nn.LayerNorm(config.hidden_size)
            self._init_weights(self.ln)

        self.classifier = nn.Linear(input_hidden_size, 1)

        self._init_weights(self.classifier)
        for mod in self.classification_head:
            self._init_weights(mod)

        if config.loss == "mse":
            self.loss_fct = nn.MSELoss()
        elif config.loss == "bce":
            self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs,
    ):

        token_type_ids = (
            {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **token_type_ids,
            **kwargs,
        )

        for i, mod in enumerate(self.classification_head):
            if i == 0:
                if "COCO" in str(self.config.__class__):
                    hs = outputs[1]
                else:
                    hs = outputs.hidden_states
                x = mod(hs, attention_mask=attention_mask)
            else:
                x = mod(x, attention_mask=attention_mask)

        loss = None
        if labels is not None:

            if self.config.multisample_dropout:
                layer_nm = None
                if self.config.output_layer_norm:
                    layer_nm = self.ln
                loss, logits = self.multisample_dropout(x, self.classifier, labels, self.loss_fct, layer_nm)
            else:
                x = self.dropout(x)

                if self.config.output_layer_norm:
                    logits = self.classifier(self.ln(x))
                else:
                    logits = self.classifier(x)

                if self.config.loss == "mse":
                    loss = self.loss_fct(logits.sigmoid().view(-1), labels.view(-1))
                elif self.config.loss == "bce":
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))

        else:
            if self.config.output_layer_norm:
                logits = self.classifier(self.ln(x))
            else:
                logits = self.classifier(x)

        return SequenceClassifierOutput(
            loss=loss, logits=logits, probas=logits.sigmoid()
        )

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        """Initialize the weights"""
        if isinstance(module, nn.Sequential):
            for m in module.modules():
                self._init_weights(m)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def get_pretrained(config, model_path):
    model = USPPPMModel(config)

    if model_path.endswith("pytorch_model.bin"):
        model.load_state_dict(torch.load(model_path))
    elif "cocolm" in model_path:
        model.backbone = COCOLMModel.from_pretrained(
            model_path,
            config=config,
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )
    else:
        model.backbone = AutoModel.from_pretrained(
            model_path,
            config=config,
            use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN", True),
        )

    return model


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probas: torch.FloatTensor = None


class AttentionHead(nn.Module):
    """
    Weights the hidden states.
    Maybe from this: https://www.kaggle.com/code/maunish/clrp-roberta-svm?scriptVersionId=65180276&cellId=6
    https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/324330#1793635
    """

    def __init__(self, input_hidden_size) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_hidden_size, input_hidden_size),
            nn.LayerNorm(input_hidden_size),
            nn.GELU(),
            nn.Linear(input_hidden_size, 1),
        )

    def forward(self, hidden_states, attention_mask, **kwargs):
        x = self.attention(hidden_states)
        x[attention_mask == 0] = float("-inf")
        weights = torch.softmax(x, 1)

        out = torch.sum(weights * hidden_states, dim=1)

        return out


class ConcatHiddenStates(nn.Module):
    """
    Concatenates the last `num_concat` hidden states.
    """

    def __init__(self, num_concat) -> None:
        super().__init__()

        self.num_concat = num_concat

    def forward(self, all_hidden_states, **kwargs):
        return torch.cat([hs for hs in all_hidden_states[-self.num_concat :]], dim=-1)


class BiLSTMHead(nn.Module):
    """
    WIP
    """

    def __init__(self, embedding_dim, hidden_dim) -> None:
        super().__init__()

        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=1, bidirectional=True
        )

    def forward(self, x):
        x, _ = self.bilstm(x)
        return x


class CLSHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        return hidden_states[:, 0, :]


class MeanPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, attention_mask, **kwargs):
        x = hidden_states
        mask = attention_mask.unsqueeze(-1).expand(x.size())
        x = torch.sum(x * mask, 1)
        x = x / (mask.sum(1) + 1e-8)
        return x


class MaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        max_pooled, _ = torch.max(hidden_states, 1)
        return max_pooled


class MeanMaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max = MaxPoolHead()
        self.mean = MeanPoolHead()

    def forward(self, hidden_states, **kwargs):
        max_pooled = self.max(hidden_states)
        mean_pooled = self.mean(hidden_states, **kwargs)

        return torch.cat([max_pooled, mean_pooled], dim=1)


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_probs, problem_type, num_labels) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]
        self.problem_type = problem_type
        self.num_labels = num_labels

    def forward(self, hidden_states, linear, labels, loss_fn, layer_nm=None):
        if layer_nm is None:
            layer_nm = nn.Identity()
        logits = [linear(layer_nm(d(hidden_states))) for d in self.dropouts]
        if self.problem_type == "regression":
            logits = [l.view(-1) for l in logits]
            labels = labels.view(-1)
            if "MSE" in str(loss_fn.__class__):
                logits = [l.sigmoid() for l in logits]
        elif self.problem_type == "single_label_classification":
            logits = [l.view(-1, self.num_labels) for l in logits]
            labels = labels.view(-1)
        losses = [loss_fn(log, labels) for log in logits]

        logits = torch.mean(torch.stack(logits, dim=0), dim=0)
        loss = torch.mean(torch.stack(losses, dim=0), dim=0)

        return (loss, logits)


class Seq2SeqUSPPPMModel(PreTrainedModel):

    def __init__(self, config, id2score, **kwargs):
        super().__init__(config)
        self.backbone = AutoModelForSeq2SeqLM.from_config(config)

        self.id2score = id2score
        self.ids = torch.tensor(list(id2score.keys()))
        self.scores = list(id2score.values())


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):

        if labels is not None:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                )

            loss, logits = outputs.loss, outputs.logits[:, 1, :]
            preds = logits.index_select(-1, self.ids.to(input_ids.device))
            preds = preds.argmax(-1).squeeze()
            preds = torch.tensor([self.scores[x] for x in preds.detach().cpu().tolist()])
        else:
            loss = None

            output = self.backbone.generate(
                input_ids=input_ids,
                return_dict_in_generate=True, 
                output_scores=True,
                min_length=3,
                max_length=3,
                )
            logits = None
            id_scores = output.scores[1].index_select(-1, self.ids.to(input_ids.device))
            preds = id_scores.argmax(-1).squeeze()
            preds = torch.tensor([self.id2score[x] for x in preds.detach().cpu().tolist()])

        return (loss, logits, preds)

