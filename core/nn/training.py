from core.transform import transform
import core.nn as nn
from core.tensor import Tensor
from core.autograd import grad

def train(lossfn, optim, lr, train_loader, ):
    network = transform(net_fn)
    optimizer = optim
    
    def loss(params, X, y):
        out = network.apply(params, X)
        out = lossfn(out, y)
        return out

    def update(state, X, y):
        grads = grad(loss)(state.params, X, y)
        params, opt_state = optimizer.update(grads, state.opt_state)
        return nn.TrainingState(params, opt_state)
   
    def evaluate(params, X, y):
        out = network.apply(params, X, y)
        predictions = np.argmax(out, axis=-1)
        return Tensor.mean(predictions == y)

    train_loader, x_init, y_init = load_dataset()
    init_params = network.init(x_init.flatten())
    init_opt_state = optimizer.init(init_params)
    state = nn.TrainingState(params=init_params, opt_state=init_opt_state)

    for epoch in range(10):
        for X, y in train_loader:
            X = Tensor(np.array(X)).flatten()
            y = Tensor(np.array(y))
            state = update(state, X, y)

        accuracy = evaluate(state.params, X, y)
        print(f"epoch: {epoch}, accuracy: {accuracy:.3f}")
