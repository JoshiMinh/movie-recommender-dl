import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def download_and_extract_movielens(data_dir='data'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    zip_path = os.path.join(data_dir, 'ml-1m.zip')
    extract_path = os.path.join(data_dir, 'ml-1m')
    
    if not os.path.exists(extract_path):
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        print(f"Downloading Movielens 1M from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    
    return extract_path

def load_data(data_dir='data'):
    extract_path = download_and_extract_movielens(data_dir)
    
    ratings_file = os.path.join(extract_path, 'ratings.dat')
    movies_file = os.path.join(extract_path, 'movies.dat')
    
    # ML-1M dat files are :: separated
    ratings_df = pd.read_csv(ratings_file, sep='::', engine='python', 
                             names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             encoding='latin-1')
                             
    movies_df = pd.read_csv(movies_file, sep='::', engine='python',
                            names=['movie_id', 'title', 'genres'],
                            encoding='latin-1')
                            
    return ratings_df, movies_df

def process_data(ratings_df, max_seq_len=50):
    # Remap movie IDs to 1..M (0 is reserved for padding)
    unique_movies = ratings_df['movie_id'].unique()
    movie2idx = {mid: idx + 1 for idx, mid in enumerate(unique_movies)}
    
    # Remap user IDs to 0..U-1
    unique_users = ratings_df['user_id'].unique()
    user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
    
    ratings_df['movie_idx'] = ratings_df['movie_id'].map(movie2idx)
    ratings_df['user_idx'] = ratings_df['user_id'].map(user2idx)
    
    # Sort strictly chronologically
    ratings_df = ratings_df.sort_values(['user_idx', 'timestamp'])
    
    train_data, val_data, test_data = [], [], []
    
    for user_idx, user_df in ratings_df.groupby('user_idx'):
        movies = user_df['movie_idx'].tolist()
        n_inter = len(movies)
        
        if n_inter < 3: # Handle edge case: need at least 3 for valid split (train, val, test)
            continue
            
        train_end = int(n_inter * 0.8)
        val_end = int(n_inter * 0.9)
        
        # ML-1M has min 20 ratings per user, so train_end >= 16, val_end >= 18
        
        train_movies = movies[:train_end]
        val_movies = movies[train_end:val_end]
        test_movies = movies[val_end:]
        
        def get_seqs(interactions):
            seqs, targs = [], []
            for i in range(1, len(interactions)):
                # Truncate to max_seq_len correctly
                seq = interactions[max(0, i - max_seq_len):i]
                seqs.append(seq)
                targs.append(interactions[i])
            return seqs, targs
            
        t_seq, t_targ = get_seqs(train_movies)
        v_seq, v_targ = get_seqs(val_movies)
        te_seq, te_targ = get_seqs(test_movies)
        
        for s, t in zip(t_seq, t_targ):
            train_data.append((user_idx, s, t))
        for s, t in zip(v_seq, v_targ):
            val_data.append((user_idx, s, t))
        for s, t in zip(te_seq, te_targ):
            test_data.append((user_idx, s, t))
            
    return train_data, val_data, test_data, movie2idx, user2idx

class MovieSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        user_idx, seq, target = self.data[idx]
        return (torch.tensor(user_idx, dtype=torch.long), 
                torch.tensor(seq, dtype=torch.long), 
                torch.tensor(target, dtype=torch.long))

def collate_fn(batch):
    user_ids = [item[0] for item in batch]
    sequences = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    
    sequence_lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    # Right-padding with 0
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    return torch.stack(user_ids), padded_sequences, sequence_lengths, torch.stack(targets)

def get_dataloaders(batch_size=256, max_seq_len=50, data_dir='data'):
    ratings_df, movies_df = load_data(data_dir=data_dir)
    train_data, val_data, test_data, movie2idx, user2idx = process_data(ratings_df, max_seq_len=max_seq_len)
    
    train_dataset = MovieSequenceDataset(train_data)
    val_dataset = MovieSequenceDataset(val_data)
    test_dataset = MovieSequenceDataset(test_data)
    
    # Shuffle only training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Useful artifacts for modeling and demo
    num_users = len(user2idx)
    num_movies = len(movie2idx) + 1 # +1 for padding index 0
    
    return train_loader, val_loader, test_loader, num_users, num_movies, movie2idx, user2idx, movies_df
