
class Optim:
    def __init__(self, params, weight_decay=0):
        self.params = params
        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            param.grad = None

 
# https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
class SGD(Optim):
    def __init__(self, params, lr, momentum=0, dampening=0, 
                 nesterov=False, maximize=False):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize

    def step(self):
        for param in self.params:
            if self.weight_decay:
                param.grad += self.weight_decay * param
            # TO-DO: implement momentum
            '''
            if momentum:
               if nesterov
            '''
            if self.maximize:
                param += self.lr * param.grad
            else
                param -= self.lr * param.grad  

# Values for arguments:
# https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
class RMSprop(Optim):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, 
                 momentum=0, centered=False):
        super().__init__()
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered


    def step(self)


class Adam(Optim):
    def __init__(self, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, maximize=False):
        super().__init__()
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize


    def step(self):



