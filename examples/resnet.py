import AIlib.utils as utils
from AIlib.tensor import Tensor
from AIlib.autograd import grad
import AIlib.nn as nn
from AIlib.nn.module import wrap_method
import AIlib.transform as transform
# Used for dataloading
import torch
from torchvision import datasets, transforms
import tqdm

def load_dataset():
    transformn = transforms.Compose(
            [transforms.ToTensor()])
    batch_size = 256
    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = datasets.MNIST(root='./data', train=False,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False num_workers=2)

   x_init, y_init = Tensor.zeros((256, 28, 28, 3)), Tensor.zeros((256,))
   return trainloader, x_init, y_init


class ResNetBlock(nn.Module):
    def __init__(self, out_channels, stride=1, use_1x1_conv=False):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d()
        self.conv2 = nn.Conv2d(out_channels=our_channels, kernel_size=3, stride=stride, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn2 = nn.BatchNorm2d()

    @wrap_method
    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        return (out + x).relu()

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(out_channels=64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d()
        self.l1 = self._make_layer(64, 2, first_layer=True)
        self.l2 = self._make_layer(128, 2)
        self.l3 = self._make_layer(256, 2)
        self.l4 = self._make_layer(512, 2)
        self.fc = nn.Linear(10)
    
    def _make_layer(self, channels, num_blocks, first_layer=False):
        layers = []
        for b in range(num_blocks):
            if b == 0 and not first_layer:
                layers.append(ResNetBlock(channels, stride=2, use_1x1_conv=True)
            else:
            layers.append(ResNetBlock(channels))
        return layers

    @wrap_method
    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = out.sequential(self.l1)
        out = out.sequential(self.l2)
        out = out.sequential(self.l3)
        out = out.sequential(self.l4)
        out = out.mean(3).mean(2)
        out = self.fc(out)
        return out

def net_fn(x):
    return ResNet()(x)

lossfn = nn.CategoricalCrossEntropy()

def main():
    network = transform(net_fn)
    optimizer = optim.sgd(1e-3)

    def loss_fn(params, x, y):
        out = network.apply(params, x)
        loss = lossfn(out, y)
        return loss

    def update_weights(params, x, y):
        grads, loss = grad(loss_fn)(params, x, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state), loss

    trainloader, x_init, y_init = load_dataset()
    init_params = network.init(x_init, y_init)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)

    for epoch in tqdm(range(10)):
        epoch_loss = 0
        for x, y in trainloader:
            x = Tensor(np.array(x)).detach()
            y = Tensor(utils.one_hot(y, 10)).detach()
            state, loss = update_weights(state.params, x, y)
            epoch_loss += loss

        print(f"Loss on epoch: {epoch} was {epoch_loss}")


if __name__ == "__main__":
    main()

