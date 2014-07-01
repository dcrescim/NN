import numpy as np
from numpy import dot, transpose, multiply
from scipy.signal import convolve2d, correlate2d, convolve, correlate

'''
Change Sig (to the one specified by Lecun's paper)
'''
'''
  Error Functions
'''

class PureObject(object):

  def __eq__(self, other):
    return (type(self) == type(other))

class CrossEntropyError(PureObject):

  @staticmethod
  def func(Y,T):
    rows = Y.shape[0]
    first = np.sum(multiply(T, np.log(Y)))
    second = np.sum(multiply(1-T, np.log(1-Y)))
    return -(first + second)/rows

  @staticmethod
  def grad(Y,T):
    rows = Y.shape[0]
    diff = Y-T
    denom = 1.0/np.multiply(Y, 1-Y)
    return np.multiply(diff, denom)/rows

class SquaredError(PureObject):
  @staticmethod
  def func(Y,T):
    m = Y.shape[0] * 1.0 # Force into real number
    diff = Y - T
    squares = multiply(diff, diff)
    return np.sum(squares)/ (2*m)
    
  @staticmethod
  def grad(Y,T):
    rows = Y.shape[0] * 1.0# Force into real number
    diff = Y-T
    return diff/rows

class AccuracyScore(PureObject):

  @staticmethod
  def func(Y,T):
    diff = np.ravel(Y) - np.ravel(T)
    count = np.count_nonzero(diff)
    return 1 - float(count)/len(Y)


class RegressScore(PureObject):

  @staticmethod
  def func(Y,T):
    u = ((T - Y)**2).sum()
    v = ((T-T.mean())**2).sum()
    return 1 - u/v

'''
  Matrix Functions
'''

# Make faster Sig as Lecun describes in his paper.
class Sig(PureObject):
  @staticmethod
  def func(x):
    return 1/(1+np.exp(-x))

  @staticmethod
  def grad(x):
    return multiply(Sig.func(x),1 - Sig.func(x))

class Tanh(PureObject):
  @staticmethod
  def func(x):
    return np.tanh(x)

  @staticmethod
  def grad(x):
    return 4/((np.exp(x) + np.exp(-x)))**2


class HardTanh(PureObject):
  @staticmethod
  def func(x):
    x[np.where(x >  1)] =  1
    x[np.where(x < -1)] = -1
    return x

  @staticmethod
  def grad(x):
    h = np.ones(x.shape)
    h[np.where(x > 1)] = 0
    h[np.where(x < -1)] = 0
    return h


'''
Maybe make a Rect with optional parameter variance
'''

class Rect(PureObject):

  @staticmethod
  def func(x):
    return max(0, x)

  @staticmethod
  def grad(x):
    if x > 0:
      return 1
    else:
      return 0


'''
  One off helper functions
'''

def wrap_variables(*args):
  return map(lambda x: np.nditer(x, ['multi_index'], ['readwrite']), args) 

def coalesce(numpy_arr):
  pdf_array = np.zeros(numpy_arr.shape)
  maxes = numpy_arr.argmax(axis=1)
  for (index,elem) in enumerate(maxes):
    pdf_array[index,elem] = 1
  return pdf_array

# Takes numpy array, and returns a row array
def row(arr):
  if len(arr.shape) == 1:
    return arr.reshape(1, len(arr))
  return arr

def col(arr):
  if len(arr.shape) == 1:
    return arr.reshape(len(arr), 1)
  return arr











