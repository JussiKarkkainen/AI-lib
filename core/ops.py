import graph

class Ops:
    def __init__(self):
        self.value = None
        self.inputs = []
        self.grad = None
        

class Add(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"add" if name is None else name

    def forward(self, x, y):
        return x + y

    def backward():
        pass

class Mul(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"mul" if name is None else name

    def forward(self, x, y):
        return x * y

    def backward():
        pass

class Sub(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"sub" if name is None else name

    def forward(self, x, y):
        return x - y
    
    def backward():
        pass

class Div(Ops):
    def __init__(self, x, y, name=None):
        super().__init__(name)
        self.inputs = [x, y]
        self.name = f"div" if name is None else name

    def forward(self, x, y):
        return x * y**-1
    
    def backward():
        pass

class Pow(Ops):
    def __init__(self, x, y, name=None):
        self.inputs = [x, y]
        self.name = f"pow" if name is None else name

    def forwardi(self, x, y):
        return x ** y

    def backward():
        pass
