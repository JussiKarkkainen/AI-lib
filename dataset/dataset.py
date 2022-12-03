from AIlib.tensor import Tensor

class Dataset():
    def __init__(self, data):
        mod = len(data) % 32
        self.data = data[0:len(data)-mod]
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
        self.block_size = 128

    def __getitem__(self, idx):
        # Grab a chunk of characters 
        chunk = self.data[idx:idx + self.block_size + 1]
        # Encode characters to ints
        dix = [self.stoi[s] for s in chunk]
        # Return as tensors
        x = Tensor(dix[:-1])
        y = Tensor(dix[1:])
        return x, y

