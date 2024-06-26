import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LinguisticModule(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LinguisticModule, self).__init__()
        self.lf = torch.nn.Linear(input_size, hidden_size)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, ling_features):
        outputs = self.lf(ling_features)
        outputs = F.leaky_relu(outputs)
        outputs = self.dropout(outputs)

        return outputs

class SemanticModule(nn.Module):
    def __init__(self, essay_encoder, unfreeze, dropout=0.1):
        super(SemanticModule, self).__init__()

        if unfreeze == 'none':
            for param in essay_encoder.parameters():
                param.requires_grad = False
        elif unfreeze == 'all':
            pass
        elif unfreeze == 'encoder':
            for param in essay_encoder.parameters():
                param.requires_grad = False
            for param in essay_encoder.encoder.parameters():
                param.requires_grad = True
        else: # unfreeze pooler by default
            for param in essay_encoder.parameters():
                param.requires_grad = False
            for param in essay_encoder.pooler.parameters():
                param.requires_grad = True

        self.essay_encoder = essay_encoder
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, token_ids, attention_mask):
        outputs = self.essay_encoder(
            token_ids,
            attention_mask=attention_mask)

        pooled_outputs = outputs[1]
        pooled_outputs = self.dropout(pooled_outputs)

        return pooled_outputs
    
class CoherenceModule(nn.Module):
    def __init__(self, sentence_encoder, unfreeze, dropout=0.1):
        super(CoherenceModule, self).__init__()
        
        if unfreeze == 'none':
            for param in sentence_encoder.parameters():
                param.requires_grad = False
        elif unfreeze == 'all':
            pass
        elif unfreeze == 'encoder':
            for param in sentence_encoder.parameters():
                param.requires_grad = False
            for param in sentence_encoder.encoder.parameters():
                param.requires_grad = True
        else: # unfreeze pooler by default
            for param in sentence_encoder.parameters():
                param.requires_grad = False
            for param in sentence_encoder.pooler.parameters():
                param.requires_grad = True
        self.sentence_encoder = sentence_encoder

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.sentence_encoder(
            input_ids,
            attention_mask=attention_mask)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        return pooled_output


class LSCModel(nn.Module):
    def __init__(self, essay_encoder, essay_unfreeze: str | None,
                 sentence_encoder, sentence_unfreeze: str | None,
                 input_lf_size, hidden_lf_size = 64,  hidden_list: List | None = 0, 
                 dropout=0.1):
        super(LSCModel, self).__init__()
        self.linguistic_module = LinguisticModule(
            input_lf_size, hidden_lf_size)
        self.semantic_module = SemanticModule(essay_encoder, essay_unfreeze, dropout=dropout)
        self.coherence_module = CoherenceModule(sentence_encoder, sentence_unfreeze, dropout=dropout)
        
        # self.fc = nn.Linear(
        #     hidden_lf_size + essay_encoder.config.hidden_size + sentence_encoder.config.hidden_size, 768)

        # self.regressor = nn.Linear(768, 1)

        last_hidden_size = hidden_lf_size + essay_encoder.config.hidden_size + sentence_encoder.config.hidden_size

        self.hidden_list = hidden_list

        for i, hidden_size in enumerate(hidden_list):
            setattr(self, f'fc{i}', nn.Linear(last_hidden_size, hidden_size))
            setattr(self, f'fc{i}_relu', nn.LeakyReLU())
            setattr(self, f'fc{i}_dropout', nn.Dropout(dropout))

            last_hidden_size = hidden_size

        self.regressor = nn.Linear(last_hidden_size, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ling_features, essay_token_ids, essay_attention_mask,
                sentence_token_ids, sentence_attention_mask):
        ling_outputs = self.linguistic_module(ling_features)

        sem_outputs = self.semantic_module(essay_token_ids, essay_attention_mask)

        coh_outputs = self.coherence_module(sentence_token_ids.view(-1, sentence_token_ids.shape[-1]), 
                                            sentence_attention_mask.view(-1, sentence_attention_mask.shape[-1]))

        coh_infor = coh_outputs.view(-1, sentence_token_ids.size(1), coh_outputs.size(-1))
        coh_infor, _ = torch.max(coh_infor, dim=1)

        outputs = torch.cat([ling_outputs, sem_outputs, coh_infor], dim=-1)

        for i in range(len(self.hidden_list)):
            outputs = getattr(self, f'fc{i}')(outputs)
            outputs = getattr(self, f'fc{i}_relu')(outputs)
            outputs = getattr(self, f'fc{i}_dropout')(outputs)

        outputs = self.regressor(outputs)

        return outputs
