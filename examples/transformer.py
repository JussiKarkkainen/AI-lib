from dataset.loader import load_tiny_shakespeare
from AIlib.tensor import Tensor
from AIlib.autograd import grad
from AIlib.transform import transform
from AIlib.nn.module import wrap_method
import AIlib.nn as nn

def get_dataset():
    dataset = load_tiny_shakespeare()
    print(len(dataset.data))
    return dataset, dataset[0]


class CausalSelfAttention(nn.Module):
    '''
    MultiHeadAttention?
    '''
    pass

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        sefl.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = [nn.Linear(4*config.n_embed), nn.Linear(config.n_embed)]

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = self.ln2(x).sequential(self.mlp).gelu().dropout(config.dropout)
        return x

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    @wrap_method
    def __call__(self, x):
        out = encoder(x)
        return decoder(out)

def net_fn(x):
    net = Transformer()
    return net(x)

def main():
    network = transform(net_fn)
    optimizer = nn.optim.sgd(1e-3)

    def loss_fn(params, X, y):
        pass

    def update(params, X, y):
        pass


    train_loader, X_init = get_dataset()


if __name__ == "__main__":
    main()
