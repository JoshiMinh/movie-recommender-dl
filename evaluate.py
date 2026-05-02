import torch

def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluates the trained SequenceRecommender on Test split interactions to compute:
    - Top-1 Accuracy
    - Hit Ratio @ 10
    """
    model.eval()
    
    top1_correct = 0
    hr10_hits = 0
    total = 0
    
    with torch.no_grad():
        for user_ids, padded_seqs, seq_lengths, targets in test_loader:
            user_ids = user_ids.to(device)
            padded_seqs = padded_seqs.to(device)
            targets = targets.to(device)
            
            # Application of Softmax dimension matching inside predict()
            probs = model.predict(user_ids, padded_seqs, seq_lengths)
            
            # 1. Top-1 Accuracy: does the argmax match the target?
            top1_preds = probs.argmax(dim=-1)
            top1_correct += (top1_preds == targets).sum().item()
            
            # 2. Hit Ratio @ 10: does the target exist in the top 10 probabilities?
            top10_preds = torch.topk(probs, k=10, dim=-1).indices # shape: (batch_size, 10)
            
            # Expanding target for comparison Broadcast matching
            targets_expanded = targets.unsqueeze(1).expand_as(top10_preds)
            hits = (top10_preds == targets_expanded).sum().item()
            hr10_hits += hits
            
            total += targets.size(0)
            
    top1_acc = top1_correct / total if total > 0 else 0
    hr10 = hr10_hits / total if total > 0 else 0
    
    return top1_acc, hr10
