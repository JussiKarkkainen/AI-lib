from dataset.loader import load_tiny_shakespeare
from AIlib.tensor import Tensor
from AIlib.autograd import grad
from AIlib.transform import transform
from AIlib.nn.module import wrap_method
import AIlib.nn as nn
from typing import NamedTuple
import numpy as np
from tqdm import tqdm

def get_dataset(batch_size=32):
    dataset = load_tiny_shakespeare()
    return dataset, dataset[0]

class Config(NamedTuple):
    n_layers = 3
    n_head = 3
    n_embed = 48
    vocab_size = 65
    block_size = 128
    dropout = 0.1

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask = self.subsequent_mask()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = nn.MultiheadAttention(config.n_head, config.n_embed, dropout=0.1, mask=self.mask)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = [nn.Linear(4*config.n_embed), nn.GELU(), nn.Linear(config.n_embed)]
    
    def subsequent_mask(self):
        mask = Tensor.tril(Tensor.ones((self.config.block_size, self.config.block_size))).reshape((1, 1, self.config.block_size, self.config.block_size))
        return mask

    @wrap_method
    def __call__(self, x):
        ln1 = self.ln_1(x)
        x = x + self.attn(ln1, ln1, ln1)
        x = x + self.ln_2(x).sequential(self.mlp).dropout(self.config.dropout)
        return x

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.tok_embed = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.pos_embed = nn.Embedding(self.config.block_size, self.config.n_embed)
        self.block = [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        self.ln = nn.LayerNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.vocab_size, bias=False)

    @wrap_method
    def __call__(self, x):
        b, t = x.shape
        pos = Tensor(np.expand_dims(np.arange(0, t), 0))
        tok_embed = self.tok_embed(x)
        pos_embed = self.pos_embed(x)
        x = (tok_embed + pos_embed).dropout(prob=0.1)
        x = x.sequential(self.block)
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits

def net_fn(x):
    net = Transformer()
    return net(x)

lossfn = nn.CategoricalCrossEntropyLoss()

def main():
    network = transform(net_fn)
    optimizer = nn.optim.sgd(1e-3)

    def loss_fn(params, X, y):
        out = network.apply(params, X)
        loss = lossfn(out, y)
        return loss

    def update_weights(params, X, y):
        grads, loss = grad(loss_fn)(params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state), loss

    train_loader, (X_init, y_init) = get_dataset()
    X_init = Tensor(np.expand_dims(X_init.data, 0))
    init_params = network.init(X_init)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)
    
    print("Starting Training")
    for epoch in range(10):
        epoch_loss = 0
        for X, y in tqdm(train_loader):
            X = Tensor(np.expand_dims(X.data, 0)).detach()
            y = Tensor(np.expand_dims(y.data, 0))
            y = nn.utils.one_hot(y, Config().vocab_size).detach()
            state, loss = update_weights(state.params, X, y)
            epoch_loss += loss
    
        print(f"Loss on Epoch: {epoch} was {epoch_loss}")

if __name__ == "__main__":
    main()
