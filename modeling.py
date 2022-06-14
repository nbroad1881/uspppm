import os
from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
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
            self.multisample_dropout = MultiSampleDropout(config.multisample_dropout)

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
                x = mod(outputs.hidden_states, attention_mask=attention_mask)
            else:
                x = mod(x, attention_mask=attention_mask)

        loss = None
        if labels is not None:

            if self.config.multisample_dropout:
                x = self.multisample_dropout(x)
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
    def __init__(self, dropout_probs) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]

    def forward(self, x):
        return torch.mean(
            torch.stack([dropout(x) for dropout in self.dropouts], dim=0), dim=0
        )
