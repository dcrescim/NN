import numpy as np
from numpy import dot, transpose, multiply
from scipy.signal import convolve2d, correlate2d, convolve, correlate
from skimage.measure import block_reduce
from functions import *
import operator
'''
  Ideas for other layers
  0. SoftMaxLayer
  1. Reshape layer (Function on the whole ndarray)
  2. Max Layer (Function on blocks)
  3. Block Func Layer (Function on local blocks)
  4. Faster Sig Layer(Function on every element)
  5. Functions on random subset (Dropout like functions)

  Alternative Implementation of Layers
'''


# Covers HardTanh, Tanh, Sig, Rect
class FunctionLayer(PureObject):
  def __init__(self, mapping):
    self.mapping = mapping

  def forward(self, X):
    self.X = X
    return self.mapping.func(self.X)

  def north_partial(self, partials):
    return []

  def west_partial(self, partials):
    return partials*self.mapping.grad(self.X)

  def delta_iterator(self):
    return []

  def __repr__(self):
    return "Function Layer: Func = " + str(self.mapping)

class HardTanhLayer(FunctionLayer):
  def __init__(self):
    super(HardTanhLayer, self).__init__(mapping=HardTanh)

class TanhLayer(FunctionLayer):
  def __init__(self):
    super(TanhLayer, self).__init__(mapping=Tanh)

class RectLayer(FunctionLayer):
  def __init__(self):
    super(RectLayer, self).__init__(mapping=Rect)

class SigLayer(FunctionLayer):
  def __init__(self):
    super(SigLayer, self).__init__(mapping=Sig)

# Covers mean layer
class MeanLayer(PureObject):
  def __init__(self, dim):
    self.dim = dim
    self.units_per_block = float(reduce(operator.mul, dim))

  def forward(self, X):
    self.X = X
    return block_reduce(self.X, block_size=self.dim, func=np.mean)

  def north_partial(self, partials):
    return []

  # Need to check this
  def west_partial(self, partials):
    return np.kron(partials, np.ones(self.dim))/ self.units_per_block

  def get_params(self, deep=True):
    return {'dim': self.dim}

  # Writes actual contents. No copies.
  def set_params(self, **params):
    for parameter, value in params.items():
      self.__setattr__(parameter, value)

  def delta_iterator(self):
    return []

  def __repr__(self):
    return "Mean Pooling Layer: Dim = " + str(self.dim)

'''
class SoftMaxLayer(PureObject):
  pass
'''
class DotLayer(object):
  def __init__(self, dim, W = None, b = None):
    if W is None:
      a = 1. / dim[0]
      self.W = np.array(np.random.uniform(
        low = -a,
        high = a,
        size = dim
      ))
    else:
      self.W = np.copy(W)

    if b is None:
      a = 1. / dim[0]
      self.b = np.array(np.random.uniform(
        low = -a,
        high = a,
        size = (1, dim[1])
      ))

    else:
      self.b = np.copy(b)

    self.dim = dim
  def forward(self, X):
    self.X = X
    return np.dot(X,self.W) + self.b

  def north_partial(self, partials):
    W_grad = np.dot(self.X.T, partials)
    b_grad = np.sum(partials, axis=0)
    
    self.W -= W_grad
    self.b -= b_grad
    return [W_grad, b_grad]

  def west_partial(self, partials):
    return np.dot(partials, self.W.T)

  def delta_iterator(self):
    layer_params = [self.W, self.b]
    return map(lambda x: np.nditer(x, ['multi_index'], ['readwrite']), layer_params)

  # Returns actual contents. No copies.
  def get_params(self, deep=True):
    return {'W' : self.W, 'b' : self.b, 'dim' : self.dim}

  # Writes actual contents. No copies.
  def set_params(self, **params):
    for parameter, value in params.items():
      self.__setattr__(parameter, value)

  def __eq__(self, other):
    return ((type(self) == type(other)) and
            (np.allclose(self.W, other.W)) and
            (np.allclose(self.b, other.b))
           )

  def __repr__(self):
    return "DotLayer W dim= %s b dim=%s" % (str(self.W.shape), str(self.b.shape))

'''
Experimental Layers
'''

'''
Simple Convolution Layer dim=2D, numb_kernels = n
'''

class ConvLayer2D:
  def __init__(self, dim, numb_kernels=1, W=None, b=None):
    # 2-D supported
    assert len(dim) == 2, "dimension is not 2D"

    if W is None:
      a = 1. / dim[0]
      self.W = []
      for i in range(numb_kernels):
        cur = np.array(np.random.uniform(
          low = -a,
          high = a,
          size = dim
        ))
        self.W.append(cur)
    
    else:
      self.W = np.copy(W)

    # The intercept is just one list
    if b is None:
      a = 1. / dim[0]
      self.b = np.random.uniform(low=-a, high=a, size=numb_kernels)
      
    else:
      self.b = np.copy(b)

    self.numb_kernels = numb_kernels
    self.dim = dim

  def forward(self, X):
    self.X = X
    self.output = np.zeros((self.numb_kernels, X.shape[0] - self.dim[0] +1, X.shape[1] - self.dim[1] + 1))
    # I use scipy correlate instead of convolve, 
    # because it doesn't rotate my matrix
    for i in range(self.numb_kernels):
      self.output[i] = correlate2d(X, self.W[i], mode='valid') + self.b[i]
  
    return self.output

  def north_partial(self, partials):
    W_partial = []
    b_partial = []

    for i in range(self.numb_kernels):
      cur_W_partial = correlate2d(self.X, partials[i], mode='valid')
      cur_b_partial = np.sum(partials[i]) # There is only one b, not a vector
      self.W -= cur_W_partial
      self.b -= cur_b_partial
      W_partial.append(cur_W_partial)
      b_partial.append(cur_b_partial)

    return [W_partial, b_partial]

  def west_partial(self, partials):
    errors = np.zeros(self.X.shape)
    for i in range(self.numb_kernels):
      errors += convolve2d(partials[i], self.W[i], mode='full')

    return errors

  def delta_iterator(self):
    layer_params = []
    for i in range(self.numb_kernels):
      layer_params.append(self.W[i])
    layer_params.append(self.b)
    return map(lambda x: np.nditer(x, ['multi_index'], ['readwrite']), layer_params)


'''
Simple Convolution Layer dim=3D, numb_kernels=1
'''

class ConvLayer3:
  def __init__(self, dim, W=None, b=None):
    # 3-D supported
    assert len(dim) == 3, "dimension is not 3D"

    if W is None:
      a = 1. / dim[0]
      self.W = np.array(np.random.uniform(
          low = -a,
          high = a,
          size = dim
        ))
    else:
      self.W = np.copy(W)

    # The intercept is just one list
    if b is None:
      a = 1. / dim[0]
      self.b = np.random.uniform(low=-a, high=a, size=1)
      
    else:
      self.b = np.copy(b)

    self.numb_kernels = 1
    self.dim = dim

  def forward(self, X):
    self.X = X
    return correlate(self.X,self.W,mode='valid')[:,:,0] + self.b

  def north_partial(self, partials):
    W_partial = correlate(self.X, partials[:, :, np.newaxis], mode='valid')
    b_partial = np.sum(partials)
    self.W -= W_partial
    self.b -= b_partial
    return (W_partial, b_partial)

  def west_partial(self, partials):
    return convolve(partials[:, :, np.newaxis], self.W, mode='full')

  def delta_iterator(self):
    layer_params = [self.W, self.b]
    return map(lambda x: np.nditer(x, ['multi_index'], ['readwrite']), layer_params)

'''
Complex 3D Conv Layer
'''

class ConvLayer4:
  def __init__(self, dim, numb_kernels = 10, W=None, b=None):
    # 3-D supported
    assert len(dim) == 3, "dimension is not 3D"

    if W is None:
      a = 1. / dim[0]
      self.W = []
      for i in range(numb_kernels):
        cur = np.array(np.random.uniform(
            low = -a,
            high = a,
            size = dim
          ))
        self.W.append(cur)
    else:
      self.W = np.copy(W)

    # The intercept is just one list
    if b is None:
      a = 1. / dim[0]
      self.b = np.random.uniform(low=-a, high=a, size=numb_kernels)
      
    else:
      self.b = np.copy(b)

    self.numb_kernels = numb_kernels
    self.dim = dim

  def forward(self, X):
    self.X = X
    self.output = np.zeros((self.numb_kernels, X.shape[0] - self.dim[0] +1, X.shape[1] - self.dim[1] + 1))
    # I use scipy correlate instead of convolve, 
    # because it doesn't rotate my matrix
    for i in range(self.numb_kernels):
      self.output[i] = correlate(X, self.W[i], mode='valid')[:,:,0] + self.b[i]
  
    return self.output

  def north_partial(self, partials):
    W_partial = []
    b_partial = []

    for i in range(self.numb_kernels):
      cur_W_partial = correlate(self.X, partials[i][:,:,np.newaxis], mode='valid')
      cur_b_partial = np.sum(partials[i]) # There is only one b, not a vector
      W_partial.append(cur_W_partial)
      b_partial.append(cur_b_partial)

    return (W_partial, b_partial)

  def west_partial(self, partials):
    errors = np.zeros(self.X.shape)
    for i in range(self.numb_kernels):
      errors += convolve(partials[i][:,:,np.newaxis], self.W[i], mode='full')

    return errors

  def delta_iterator(self):
    layer_params = []
    for i in range(self.numb_kernels):
      layer_params.append(self.W[i])
    layer_params.append(self.b)
    return map(lambda x: np.nditer(x, ['multi_index'], ['readwrite']), layer_params)

