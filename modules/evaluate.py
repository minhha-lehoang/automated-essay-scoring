import torch

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
        for token_ids, attention_mask, features, score in dataloader:
            output = model(token_ids.to(device),
                           attention_mask.to(device),
                           features.to(device))

            loss = criterion(output, score.to(device)).float()

#             print(loss)

            running_loss += loss.item()
            all_scores.extend(score.cpu().numpy())
            predictions.extend(output.cpu().numpy())

#             break

    return running_loss / len(dataloader), torch.tensor(all_scores), torch.tensor(predictions)