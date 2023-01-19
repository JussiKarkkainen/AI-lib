from core.transform import transform
import core.nn as nn
from core.tensor import Tensor
from core.autograd import grad

def train(net_fn, optim, dataset_loader, num_epochs, one_hot=False):
    network = transform(net_fn)
    optimizer = optim
    
    def loss_fn(params, X, y):
        out = network.apply(params, X)
        out = lossfn(out, y)
        return out

    def update_weights(params, X, y):
        grads = grad(loss)(params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state), loss
   
    train_loader, x_init = dataset_loader()
    init_params = network.init(x_init)
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X, y in train_loader:
            X = Tensor(np.array(X)).flatten()
            y = nn.utils.one_hot(Tensor(np.array(y)), 10).detach() if one_hot else Tensor(np.array(y))
            state, loss = update_weights(state.params, X, y)
            epoch_loss += loss
        print(f"epoch: {epoch}, loss: {epoch_loss:.3f}")
