from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from math import sqrt
import numpy as np
from numpy import dot, transpose, multiply
from layers import *
from functions import *
import ipdb
import copy
import pickle
epsilon = 10**(-8)


class NNBase(BaseEstimator):

  def __init__(self,layers = [], lr=0.01, epochs=None, noisy=None, verbose=False):
    self.layers = copy.deepcopy(layers)
    self.lr = lr
    self.epochs = epochs
    self.noisy = noisy
    self.verbose = verbose

  def add_layer(self, layer):
    self.layers.append(layer)

  # Returns an output which matches the internal algorithm
  # Returns real numbers for regression, and probabilities for classification
  def _predict(self, X):
    current_results = X
    for layer in self.layers:
      current_results = layer.forward(current_results)

    return current_results

  def _update(self, X, T):
    if self.noisy:
      X += self.noisy*np.random.standard_normal(X.shape)
    
    Y = self._predict(X)
    cur_partial = self.error_func.grad(Y,T)*self.lr
    rev_layers = reversed(self.layers)
    for (index,layer) in enumerate(rev_layers):
      next_partial = layer.west_partial(cur_partial)
      layer.north_partial(cur_partial)
      cur_partial = next_partial
      
  def list_delta_iterators(self):
    return map(lambda x: x.delta_iterator(), self.layers)

  def _analytical_gradient(self, X, T):
    Y = self._predict(X)
    cur_partial = self.error_func.grad(Y,T)
    rev_layers = reversed(self.layers)
    gradient = []
    for layer in rev_layers:

      #Compute the partial north, and west
      next_partial = layer.west_partial(cur_partial)
      layer_grad = layer.north_partial(cur_partial)      
      cur_partial = next_partial

      gradient.append(layer_grad)

    return list(reversed(gradient))

  def _numerical_gradient(self, X, T):
    Y = self._predict(X)
    J = self.error_func.func(Y,T)
    layer_iterators = self.list_delta_iterators()
    all_gradients = []
    # Loop over layers
    for layer in layer_iterators:
      layer_deltas = []
      # Loop over W,b in layer
      for weight_structure in layer:
        grad = np.zeros(weight_structure.shape)
        # Loop over elem in W (or equivalent parameter holder)
        for elem in weight_structure:
          elem[...] = elem + epsilon
          Y_up = self._predict(X)
          J_up = self.error_func.func(Y_up,T)
          elem[...] = elem - epsilon
          grad[weight_structure.multi_index] = (J_up - J)/epsilon

        layer_deltas.append(grad)

      all_gradients.append(layer_deltas)

    return all_gradients

class NN_Classifier(NNBase):

  def __init__(self,layers = [], lr=0.01, epochs=None, noisy=None, verbose=False):
    
    super(NN_Classifier, self).__init__(layers=layers, lr=lr, epochs=epochs, noisy=noisy, verbose=verbose)
    self.type = 'C'
    self.error_func = CrossEntropyError
    self.accuracy_score = AccuracyScore
    self.label_binarizer = LabelBinarizer()

  def predict(self, X):
    predictions = []
    for el in X:
      current_prediction = NNBase._predict(self, row(el))
      predictions.append(current_prediction)
    predictions = np.vstack(predictions)
    current_results = coalesce(predictions)
    return self.label_binarizer.inverse_transform(current_results)

  def predict_proba(self, X):
    predictions = []
    for el in X:
      current_prediction = NNBase._predict(self, row(el))
      predictions.append(current_prediction)
    predictions = np.vstack(predictions)
    return predictions

  def fit(self, X, T):
    T_impl = self.label_binarizer.fit_transform(T)
    if not self.epochs:
      self.epochs = 1

    for num in xrange(self.epochs):
      if self.verbose:
        print "Epoch: %d" % num
      for i in xrange(len(X)):
        NNBase._update(self, row(X[i]), row(T_impl[i]))

  def error(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    Y = self.predict_proba(X)
    return self.error_func.func(Y, T_impl)

  def score(self, X, T):
    Y = self.predict(X)
    return self.accuracy_score.func(Y,T)

  def analytical_gradient(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    return NNBase._analytical_gradient(self, X, T_impl)

  def numerical_gradient(self, X, T):
    T_impl = self.label_binarizer.transform(T)
    return NNBase._numerical_gradient(self, X, T_impl)

class NN_Regressor(NNBase):

  def __init__(self,layers = [], lr=0.01, epochs=None, noisy=None, verbose=False):
    super(NN_Regressor, self).__init__(layers=layers, lr=lr, epochs=epochs, noisy=noisy, verbose=verbose)
    self.type = 'R'
    self.error_func = SquaredError
    self.score_func = RegressScore

  def predict(self, X):
    predictions = []
    for el in X:
      current_prediction = NNBase._predict(self, row(el))
      predictions.append(current_prediction)
    predictions = np.vstack(predictions)
    return predictions

  def fit(self, X, T):
    #import ipdb; ipdb.set_trace()
    if not self.epochs:
      self.epochs = 1

    for num in xrange(self.epochs):
      if self.verbose:
        print "Epoch: %d" % num
      for i in xrange(len(X)):
        NNBase._update(self, row(X[i]), row(T[i]))

  def score(self, X, T):
    Y = self.predict(X)
    return self.score_func.func(Y, T)

  def error(self, X, T):
    Y = self.predict(X)
    #import ipdb; ipdb.set_trace()
    return self.error_func.func(Y, T)

  def analytic_gradient(self,X,T):
    return NNBase._analytical_gradient(self, X, T)

  def numerical_gradient(self,X,T):
    return NNBase._numerical_gradient(self, X, T)

