import torch
import numpy as np

def logit_to_score(logit, min_score=1, max_score=6):
    scores = torch.clamp(torch.round(logit), min_score, max_score)
    scores = scores.long()
    return scores


def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    all_scores = []
    predictions = []

    with torch.no_grad():
        for (ling_features, essay_token_ids, essay_attention_mask,
                sentence_token_ids, sentence_attention_mask, score) in dataloader:
            output = model(ling_features.to(device),
                            essay_token_ids.to(device),
                            essay_attention_mask.to(device),
                            sentence_token_ids.to(device),
                            sentence_attention_mask.to(device))

            loss = criterion(output, score.to(device)).float()

#             print(loss)

            running_loss += loss.item()
            all_scores.extend(score.cpu().numpy())
            predictions.extend(output.cpu().numpy())

#             break

    return running_loss / len(dataloader), torch.tensor(np.array(all_scores)), torch.tensor(np.array(predictions))