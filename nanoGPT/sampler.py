import torch
from torch.utils.data import Sampler
import random


class BatchShuffleSampler(Sampler):
    def __init__(self, data_source, batch_size):
        # Initialize the base class
        super().__init__(data_source)
        # Store the data source, batch size, and generator
        self.data_source = data_source
        self.batch_size = batch_size
        # Compute the number of batches
        self.num_batches = (len(self.data_source) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        # Create a list of batch indices
        batch_indices = list(range(self.num_batches))
        # Shuffle the batch indices
        random.shuffle(batch_indices)
        # For each batch index
        for batch_index in batch_indices:
            # Compute the start and end indices of the items in the batch
            start_index = batch_index * self.batch_size
            end_index = min(start_index + self.batch_size, len(self.data_source))
            # Yield the item indices in the batch
            yield from range(start_index, end_index)
    
    def __len__(self):
        # Return the total number of items
        return len(self.data_source)