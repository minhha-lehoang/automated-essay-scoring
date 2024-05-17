import torch
import wandb

def train(model, optimizer, criterion, train_dataloader, device, is_log, logging_steps=20):
    model.train()
    running_loss = 0.0

    for i, (ling_features, essay_token_ids, essay_attention_mask,
                sentence_token_ids, sentence_attention_mask, score) in enumerate(train_dataloader):

        output = model(ling_features.to(device),
                       essay_token_ids.to(device),
                       essay_attention_mask.to(device),
                       sentence_token_ids.to(device),
                       sentence_attention_mask.to(device))

        loss = criterion(output, score.to(device)).float()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

#         if i == 5:
#             break
        if is_log:
            if (i + 1) % (logging_steps) == 0 or (i + 1) == len(train_dataloader):
                wandb.log({'train_loss_steps': running_loss / (i + 1),  # type: ignore
                        'learning_rate': optimizer.param_groups[0]['lr']})

    return running_loss / len(train_dataloader)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

