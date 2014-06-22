Neural Net
==========

This package aims to provide a python implementation of a neural net. This module is heavily influenced by scikit learn and Torch 7.

This neural net is fully compliant with the Estimator API present in every scikit learn estimator. Let's take a look at a simple example.


```python

NN = NN_Regressor()
NN.add_layer(DotLayer(dim=(4,3)))
NN.add_layer()

NN.fit(X,Y)
results = NN.predict(X)
NN.error(X,Y)
``` 

Because Neural Nets are generalizations of common linear models, we can make one that acts like a Linear Regression. 


We can also make one that mimics a logistic regression

```python 
NN = NN_Classifier()
NN.add_layer(DotLayer(dim=(4,3)))
NN.add_layer(SigLayer())

NN.fit(X,Y)
results = NN.predict(X)
NN.error(X,Y)
```

Let's build a more complicated deeper network. This will be a 4-layer network.

```python
NN= NN_Classifier()
NN.add_layer(DotLayer(dim=(4,10)))
NN.add_layer(TanhLayer())
NN.add_layer(DotLayer(dim=10,3))
NN.add_layer(SigLayer())

NN.fit(X,Y)
results = NN.predict(X)
NN.error(X,Y)
```
 
There are two flavors of this neural net, the one that performs regression (NN_Regressor) and the one that performs classification (NN_Classifier). They share a ton of code, there are a few functions though that the classifier needs in order to stay compliant with sklearn classifier (predict_proba for one).















