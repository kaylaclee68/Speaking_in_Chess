# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class ParquetDataset(Dataset):
#     def __init__(self, filename):
#         self.data = pd.read_parquet(filename)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
#         sample = torch.tensor(self.data.iloc[idx].values)
#         return sample
# Import necessary libraries
import torch
import pandas as pd

# Define the ParquetDataset class
class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, filename, feature_cols, label_cols):
        self.data = pd.read_parquet(filename)
        self.feature_cols = feature_cols
        self.label_cols = label_cols

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = torch.tensor(self.data.loc[idx, self.feature_cols].values)
        labels = torch.tensor(self.data.loc[idx, self.label_cols].values)
        return features, labels