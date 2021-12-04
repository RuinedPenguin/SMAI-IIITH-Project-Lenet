from functools import cache
import numpy as np
from numpy.lib.function_base import gradient
from numpy.random.mtrand import rand
import time
# !pip3 install tabulate
from tabulate import tabulate
np.random.seed(10)

__builtin_funcs__ =  {
    'tanh' : (lambda x : 1.7159*np.tanh(2*x/3), lambda x : 1.14393*(1-np.power(np.tanh(2*x/3),2))),
    'mse' : (lambda x,y : np.mean(np.square(y - x)), lambda x,y : -np.mean(y - x, axis=1))
}

def initialize(shape):
    mu, sigma = 0, 0.1
    b_shape = (1,1,1,shape[-1]) if len(shape)==4 else (shape[-1],)
    weight = np.random.normal(mu, sigma,  shape)
    bias  = np.ones(b_shape)*0.01
    return weight, bias

def total_params(params : 'list(tuple)') -> int:
    if not params: return 0
    return np.sum([np.prod(e.shape) for e in params])

def zero_pad(input, pad):
    return np.pad(input, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))   

class BaseLayer(object):
    def __init__(self) -> None:
        super().__init__()
        self.params = None, None
        self.optimzers =  {'adam' : None }

    def compile_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
        self.beta1, self.beta2, self.epsilon, self.eta = b1, b2, epsilon, eta
        self.m_dw, self.m_db, self.v_dw, self.v_db = None, None, None, None
        def update(dW, db, t = 1):
            if self.m_dw is None or self.m_db is None or self.v_dw is None or self.v_db is None:
                self.m_dw, self.m_db, self.v_dw, self.v_db =  np.zeros(dW.shape) , np.zeros(db.shape), np.zeros(dW.shape), np.zeros(db.shape)
            self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dW
            self.m_db = self.beta1*self.m_db + (1-self.beta1)*db
            self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dW**2)
            self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)
            m_dw_corr = self.m_dw/(1-self.beta1**t)
            m_db_corr = self.m_db/(1-self.beta1**t)
            v_dw_corr = self.v_dw/(1-self.beta2**t)
            v_db_corr = self.v_db/(1-self.beta2**t)
            w = self.params[0] - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
            b = self.params[1] - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
            self.params = (w, b) 
        self.optimzers['adam'] = update


class Conv2D(BaseLayer):

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
        z = np.zeros((input.shape[0], ) + self.out_shape)
        input_pad = zero_pad(input, self.padding)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                slice = input_pad[:, h*self.stride:h*self.stride+self.kernel_shape[0], w*self.stride:w*self.stride+self.kernel_shape[1], :]
                z[:, h, w, :] = np.tensordot(slice, self.params[0], axes=([1,2,3],[0,1,2])) + self.params[1]
        self.cache = input
        return z

    def __gradients__(self, next_d):
        dinput = zero_pad(np.zeros(self.cache.shape), self.padding)
        dW = np.zeros(self.params[0].shape)
        db = np.zeros(self.params[1].shape)
        cache_padded = zero_pad(self.cache, self.padding)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s, e = (h*self.stride, h*self.stride+self.kernel_shape[0]), (w*self.stride, w*self.stride+self.kernel_shape[1])
                slice = cache_padded[:,s[0]:s[1], e[0]:e[1], :]
                dinput[:, s[0]:s[1], e[0]:e[1], :] += np.transpose(self.params[0] @ next_d[:, h, w, :].T, (3,0,1,2))
                dW += np.matmul(np.transpose(slice, (1,2,3,0)), next_d[:, h, w, :]) # (h, w, f, b) x (b, k) 
                db += np.sum(next_d[:, h, w, :], axis=0)
        return dW, db, dinput

    def __str__(self) -> str:
        return f"CONV2D{self.param_shape ,  self.params[1].shape}"


class SubSample(BaseLayer):

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
        o = np.zeros((input.shape[0], ) + self.out_shape) 
        for h in range(self.param_shape[0]):
            for w in range(self.param_shape[1]):
                slice = input[:, h*self.stride:h*self.stride+self.kernel_shape, w*self.stride:w*self.stride+self.kernel_shape, :]
                o[:, h, w, :] = np.average(slice, axis=(1,2))
        self.cache = (input, o)
        o = self.params[0] * o + self.params[1]
        assert o.shape == (input.shape[0], ) + self.out_shape
        return o

    def __gradients__(self, next_d):
        prev_input, out_ = self.cache
        db = next_d
        dW = np.sum(np.multiply(next_d, out_))
        next_d_after = next_d * self.params[0]
        dinput = np.zeros(prev_input.shape)
        for h in range(self.out_shape[0]):
            for w in range(self.out_shape[1]):
                s , e = (h*self.stride, h*self.stride+self.kernel_shape), (w*self.stride, w*self.stride+self.kernel_shape)
                da = next_d_after[:, h, w, :][:,np.newaxis,np.newaxis,:]
                dinput[:, s[0]: s[1], e[0]: e[1], :] += np.repeat(np.repeat(da, 2, axis=1), 2, axis=2)/self.kernel_shape/self.kernel_shape
        return dW, db, dinput

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
        return None, None, next_d * self.func_d(self.cache)
    
    def __str__(self) -> str:
        return f"Activation({self.name})"

class RBF(object):
    
    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (self.out_shape[0], np.product(self.in_shape))
        self.params = ( np.random.choice([-1, 1], self.param_shape) , )
        return self
    
    def __call__(self, input, label, mode = 'test'):
        if mode == 'test': return self.predict(input)
        self.cache = input , self.params[0][label, :]
        return np.sum(0.5 * np.sum((self.cache[0] - self.cache[1]) ** 2, axis = 1, keepdims=True))
    
    def predict(self, input):
        sq_diff = input[:, np.newaxis, :] - np.array([self.params[0]] * input.shape[0]) ** 2
        pred = np.argmin(np.sum(sq_diff, axis=2), axis = 1)
        return pred

    def __gradients__(self, next_d = 1):
        return None, None, next_d * (self.cache[0] - self.cache[1])
    
    def __str__(self) -> str:
        return f"RBF{self.param_shape}"

class Dense(BaseLayer):

    def __init__(self, outputs) -> None:
        super().__init__()
        self.outputs = outputs
        self.cache = None
        self.params = None, None
        self.in_shape, self.out_shape = None, None
    
    def __call__(self, input):
        self.cache = (input.reshape((input.shape[0]), np.prod(list(input.shape)[1:])), input.shape)
        return  np.matmul(self.cache[0] , self.params[0]) + self.params[1]
    
    def init_layer(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = (self.outputs, 1)
        self.param_shape = (np.product(self.in_shape), self.out_shape[0])
        self.params = initialize(self.param_shape)
        return self

    def __gradients__(self, next_d):
        return np.matmul(self.cache[0].T ,  next_d), np.sum(next_d.T, axis = 1), np.matmul(next_d, self.params[0].T).reshape(self.cache[1])
    
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
            Activation(),
            Dense(84),
            Activation(),
            RBF(10)
        ]
        self.batch_size = None
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

    def compile_adam(self, b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01):
         self.optimizer = 'adam'
         for l in self.layers:
            if isinstance(l, BaseLayer): l.compile_adam(b1= 0.9, b2 = 0.999, epsilon = 1e-8, eta = 0.01)


    def __call__(self, input, label=None, mode = 'train'):
        o = input
        for layer in self.layers:
            if isinstance(layer, RBF): o = layer(o, label, mode)
            else: o = layer(o)
        return o

    def compute_gradients(self):
        next_d = 1
        grads = {}
        for layer in reversed(self.layers):
            dW, db, next_d = layer.__gradients__(next_d)
            if dW is None and db is None: continue
            grads[layer] = (dW, db)
        return grads

    def apply_gradients(self, gradients):
        for k,v in gradients.items():
            dW, db = v
            if isinstance(k, BaseLayer): k.optimzers[self.optimizer](dW , db)
             


if __name__ == "__main__":
    model = Lenet_SMAI()
    model.summary()
    model.compile_adam()
    batch_size = 512
    itr = 10
    st = time.time()
    img, label = np.random.rand(batch_size, 32, 32, 1), np.random.randint(0, 10,(batch_size,))
    for i in range(itr):
        loss = model(img, label)
        print(loss)
        grads = model.compute_gradients()
        model.apply_gradients(grads)
    print(model(np.random.rand(batch_size, 32, 32, 1), mode='test').shape)
    print(f'took {time.time() - st} for {itr} batch steps of size {batch_size}, 1 prediction')