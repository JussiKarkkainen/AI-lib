from AIlib.tensor import Tensor

class Dataset():
    def __init__(self, data):
        self.data = data
        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for i,ch in enumerate(chars)}
        self.block_size = 128

    def __getitem__(self, idx):
        # Grab a chunk of characters 
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # Encode characters to ints
        dix = [self.stoi[s] for s in chunk]
        # Return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

