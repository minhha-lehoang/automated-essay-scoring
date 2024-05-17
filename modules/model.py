import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFeaturesModel(torch.nn.Module):
    def __init__(self, embedder,
                 lf_input_size, lf_hidden_size=64,
                 dropout=0.2):
        super(MultiFeaturesModel, self).__init__()
        # freeze
        for param in embedder.parameters():
            param.requires_grad = False
        # unfreeze the pooler
        for param in embedder.pooler.parameters():
            param.requires_grad = True

        self.embedder = embedder
        self.lf = torch.nn.Linear(lf_input_size, lf_hidden_size)
        # self.fc1 = torch.nn.Linear(lf_hidden_size + embedder.config.hidden_size, 256)
        # self.fc2 = torch.nn.Linear(256, 128)
        self.regressor = torch.nn.Linear(
            lf_hidden_size + embedder.config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def config(self):
        return {
            'embedder': self.embedder.config,
            'lf': {
                'input_size': self.lf.in_features,
                'hidden_size': self.lf.out_features
            },
            'regressor': {
                'input_size': self.regressor.in_features,
                'output_size': self.regressor.out_features
            }
        }

    def forward(self, token_ids, attention_mask, ling_features):
        embedded = self.embedder(
            token_ids, attention_mask=attention_mask, output_hidden_states=True)[1]
        if self.training:
            embedded = self.dropout(embedded)

        ling_features = self.lf(ling_features)
        ling_features = F.leaky_relu(ling_features)
        if self.training:
            ling_features = self.dropout(ling_features)

        features = torch.cat((embedded, ling_features), dim=1)

        score = self.regressor(features)
        return score
