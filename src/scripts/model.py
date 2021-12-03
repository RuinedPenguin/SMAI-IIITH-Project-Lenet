import numpy as np
from numpy.random.mtrand import rand
# !pip3 install tabulate
from tabulate import tabulate
np.random.seed(10)

__builtin_funcs__ =  {
    'tanh' : (lambda x : 1.7159*np.tanh(2*x/3), lambda x : 1.14393*(1-np.power(np.tanh(2*x/3),2))),
    'mse' : (lambda x,y : np.mean(np.square(y - x)), lambda x,y : -np.mean(y - x, axis=1))
}

def initialize(shape):
    mu, sigma = 0, 0.1
    b_shape = (shape[-1],1)
    weight = np.random.normal(mu, sigma,  shape)
    bias  = np.ones(b_shape)*0.01
    return weight, bias

def total_params(params : 'list(tuple)') -> int:
    if not params: return 0
    return np.sum([np.prod(e.shape) for e in params])

class Conv2D(object):

    def __init__(self, num_filters, kernel_shape, stride = 1, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.in_shape, self.out_shape = None, None
        self.kernel_shape, self.num_filters = kernel_shape, num_filters
        self.params = None, None
        self.padding, self.stride = padding, stride
    
    def init_layer(self, in_shape):
        assert len(in_shape) == 3
        self.in_shape = in_shape
        self.out_shape = (in_shape[0] - self.kernel_shape[0] + 1, in_shape[1] - self.kernel_shape[1] + 1 , self.num_filters)
        self.param_shape = self.kernel_shape + (self.in_shape[-1], self.num_filters)
        self.params = initialize(self.param_shape)
        return self

    def __call__(self, input):
        pass

    def __gradients__(self, next_d):
        pass

    def __str__(self) -> str:
        return f"CONV2D{self.param_shape ,  self.params[1].shape}"


class SubSample(object):

    def __init__(self, kernel_shape, stride = 2, padding = 0) -> None:
        super().__init__()
        self.cache = None
        self.kernel_shape = kernel_shape
        self.params = None, None
        self.padding, self.stride = padding, stride
        self.in_shape, self.out_shape = None, None
    
    def init_layer(self, in_shape):
        assert len(in_shape) == 3
        self.in_shape = in_shape
        self.out_shape = ( (in_shape[0] - self.kernel_shape) // self.stride + 1, (in_shape[1] - self.kernel_shape + 1) //self.stride + 1 , in_shape[-1] )
        self.param_shape = (self.kernel_shape, self.kernel_shape) + (self.in_shape[-1], )
        self.params = np.random.rand(self.in_shape[-1], ),  np.random.rand(self.in_shape[-1], )
        return self

    def __call__(self, input):
        pass

    def __gradients__(self, next_d):
        pass

    def __str__(self) -> str:
        return f"SubSample{self.param_shape, self.params[1].shape}"

class Activation(object):

    def __init__(self, name = 'tanh') -> None:
        super().__init__()
        self.name = name
        self.cache = None
        self.params = None
        self.in_shape, self.out_shape = None, None
        self.func, self.func_d = __builtin_funcs__[name]
    
    def __call__(self, input):
        output, self.cache = self.func(input), input
        return output
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = in_shape
        return self

    def __gradients__(self, next_d):
        return next_d * self.func_d(self.cache)
    
    def __str__(self) -> str:
        return f"Activation({self.name})"

class RBF(object):
    
    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def __call__(self, input):
        self.cache = input
        return 0.5 * np.sum((self.cache - self.params[0]) ** 2, axis = 1) + self.params[1]
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (np.product(self.in_shape), self.out_shape[0])
        self.params = initialize(self.param_shape)
        return self

    def __gradients__(self, next_d):
        return -next_d * (self.cache - self.params[0]), next_d, np.sum(next_d * (self.cache - self.params[0]), axis=0)
    
    def __str__(self) -> str:
        return f"RBF{self.param_shape, self.params[1].shape}"

class Dense(object):

    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def __call__(self, input):
        self.cache = input.reshape(1, -1)
        return (self.cache @ self.params[0]).T + self.params[1]
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (np.product(self.in_shape), self.out_shape[0])
        self.params = initialize(self.param_shape)
        return self

    def __gradients__(self, next_d):
        return (next_d @ self.cache.T).T, next_d, next_d.T @ self.params[0].T
    
    def __str__(self) -> str:
        return f"Dense{self.param_shape, self.params[1].shape}"


class Lenet_SMAI(object):

    def __init__(self, input_shape = (32, 32, 1) ,name = 'Lenet') -> None:
        super().__init__()
        assert input_shape != None
        self.name = name
        self.layers = [
            Conv2D(6, (5,5)),
            SubSample(2),
            Activation(),
            Conv2D(16, (5,5)),
            SubSample(2),
            Activation(),
            Conv2D(120, (5,5)),
            Dense(84),
            Activation(),
            RBF(10)
        ]
        self.input_shape = input_shape
        prev_input_shape = input_shape
        for layer in self.layers:
           prev_input_shape = layer.init_layer(prev_input_shape).out_shape
    
    def summary(self):
        print(f"\t\t-------{self.name}-------\t\t")
        table = []
        total = 0
        for layer in self.layers:
            t = total_params(layer.params)
            table.append((layer.in_shape, str(layer), layer.out_shape, t))
            total += t
        print(tabulate(table, headers=["in", "Name (weight, bias) ", "out", "total_params"], tablefmt="psql"))
        print(f"Total Number of parameters = {total}")

    def compile(self, loss_func = 'mse', lr = 0.01):
        self.loss_func = __builtin_funcs__[loss_func]
        self.lr = lr

    def __call__(self, input, batch_size = 32):
        assert input.shape
        pass

    def compute_gradients(self):
        pass

    def compute_loss(self):
        pass

    def apply_gradients(self):
        pass

if __name__ == "__main__":
    model = Lenet_SMAI()
    model.summary()
    model.compile()
    # print(model(np.random.rand(512, 32, 32, 1), 32))