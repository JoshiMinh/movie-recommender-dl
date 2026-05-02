import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import collate_fn

def train_with_oom_fallback(create_model_fn, train_dataset, val_dataset, 
                            optimizer_name='adam', lr=1e-3, num_epochs=10, 
                            device='cuda', start_batch_size=256):
    """
    Trains the SequenceRecommender model with automatic OOM recovery by halving batch size.
    """
    batch_size = start_batch_size
    
    while batch_size >= 16:
        print(f"Starting training with batch_size={batch_size}, optimizer={optimizer_name}, lr={lr}")
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            
            model = create_model_fn().to(device)
            # Use CrossEntropyLoss directly on raw logits (per requirements)
            criterion = nn.CrossEntropyLoss()
            
            if optimizer_name.lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name.lower() == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else:
                raise ValueError("optimizer_name must be 'adam' or 'sgd'")
            
            train_losses, val_losses = [], []
            
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                
                for user_ids, padded_seqs, seq_lengths, targets in train_loader:
                    user_ids = user_ids.to(device)
                    padded_seqs = padded_seqs.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()
                    logits = model(user_ids, padded_seqs, seq_lengths)
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                avg_train = epoch_loss / len(train_loader)
                
                # Validation loop
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for user_ids, padded_seqs, seq_lengths, targets in val_loader:
                        user_ids = user_ids.to(device)
                        padded_seqs = padded_seqs.to(device)
                        targets = targets.to(device)
                        
                        logits = model(user_ids, padded_seqs, seq_lengths)
                        loss = criterion(logits, targets)
                        val_loss += loss.item()
                        
                avg_val = val_loss / len(val_loader)
                
                train_losses.append(avg_train)
                val_losses.append(avg_val)
                print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
                
            return model, train_losses, val_losses, batch_size
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                torch.cuda.empty_cache()
                print(f"CUDA OOM detected at batch_size={batch_size}. Halving batch size...")
                batch_size //= 2
            else:
                raise e
                
    raise RuntimeError("Failed to train without OOM even at minimum batch size.")
