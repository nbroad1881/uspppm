from typing import Optional
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModel,
)
from transformers.utils import ModelOutput


class USPPPMModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = AutoModel.from_config(config)

        self.dropout = nn.Dropout(config.output_dropout_prob)

        self.classification_head = [ConcatHiddenStates(config.num_concat)]
        input_hidden_size = config.hidden_size * config.num_concat

        if config.meanmax_pooling:
            self.classification_head.append(MeanMaxPoolHead())
        elif config.mean_pooling:
            self.classification_head.append(MeanPoolHead())
        elif config.max_pooling:
            self.classification_head.append(MaxPoolHead())
        else:
            self.classification_head.append(CLSHead())

        if config.attention_head:
            self.classification_head.append(
                AttentionHead(
                    input_hidden_size=input_hidden_size,
                    output_hidden_dim=config.output_hidden_dim,
                )
            )

        self.classifier = nn.Linear(config.hidden_size * config.num_concat, 1)

        self._init_weights(self.classifier)
        for m in self.classification_head.modules():
            self._init_weights(m)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs
    ):

        token_type_ids = (
            {"token_type_ids": token_type_ids} if token_type_ids is not None else {}
        )
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **token_type_ids,
            **kwargs
        )

        for i, mod in enumerate(self.classification_head):
            if i == 0:
                x = mod(outputs.hidden_states)
            else:
                x = mod(x)

        loss = None
        if labels is not None:

            if self.config.multisample_dropout:
                logits = self.classifier(self.multisample_dropout(x))
            else:
                logits = self.classifier(self.dropout(x))

            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        else:
            logits = self.classifier(x)

        return SequenceClassifierOutput(loss=loss, logits=x, probas=x.sigmoid())

    def _init_weights(self, module):
        std = self.config.to_dict().get("initializer_range", 0.02)
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
    else:
        model.backbone = AutoModel.from_pretrained(model_path)

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
    """

    def __init__(self, input_hidden_size, output_hidden_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_hidden_size, output_hidden_dim)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(output_hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden_states):
        x = self.linear1(hidden_states)
        x = self.act_fn(x)
        x = self.linear2(x)
        weights = self.softmax(x)

        out = torch.sum(weights * hidden_states, dim=1)

        return out


class ConcatHiddenStates(nn.Module):
    """
    Concatenates the last `num_concat` hidden states.
    """

    def __init__(self, num_concat) -> None:
        super().__init__()

        self.num_concat = num_concat

    def forward(self, all_hidden_states):
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

    def forward(self, hidden_states):
        return hidden_states[:, 0, :]


class MeanPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states):
        return torch.mean(hidden_states, 1)


class MaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states):
        _, max_pooled = torch.max(hidden_states, 1)
        return max_pooled


class MeanMaxPoolHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, hidden_states):
        _, max_pooled = torch.max(hidden_states, 1)
        mean_pooled = torch.mean(hidden_states, 1)

        return torch.cat([max_pooled, mean_pooled], dim=1)


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_probs) -> None:
        super().__init__()

        self.dropouts = [nn.Dropout(p=p) for p in dropout_probs]

    def forward(self, x):
        return torch.mean(
            torch.stack([dropout(x) for dropout in self.dropouts], dim=0), dim=0
        )
