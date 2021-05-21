#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

# general imports
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from IPython.display import clear_output
from funct import *

# activation functions
from numpy import tanh
from scipy.special import softmax


def derivative(f):
  """
  When f is an activation function, returns derivative w.r.t. potential of activation
  When f is a loss, returns derivative w.r.t. activation
  When f is cross_entropy and activation of output units is softmax, maths say derivative of loss w.r.t potential is one returned
  """
  if f == tanh: return dtanh
  elif f == relu: return drelu
  elif f == ide or f == softmax: return dide 
  elif f == squared_error or f == cross_entropy: return dloss


def conv_str_func(f):
  """
  if given a string of a function it gives you the actual function.
  if given the function it gives the string
  es: conv_str_func('relu')==relu()
  es: conv_str_func(relu)=='relu'
  """
  functions=[ relu , tanh , ide , softmax , squared_error , cross_entropy ]
  strings = ['relu','tanh','ide','softmax','squared_error','cross_entropy']
  for i in range(len(functions)):
    if f==functions[i]: return strings[i]
    if f==strings[i]: return functions[i]
  raise Exception('Input not valid')


class MLP():

  def __init__(self, Nodes, f='tanh', f_out='ide'):
    """
    Nh: number of hidden units for each layer
    Nu: number of input units
    Ny: number of output units
    f: activation function of hidden units
    f_out: activation function of output units
    w_range: initial range of values for entries in weight matrices
    w_range: initial number of decimals of values for entries in weight matrices
    loss: loss functions
    error: error function
    """ 
      
    if f_out=='ide':
      loss=squared_error
      error=self.MSE
    elif f_out=='softmax':
      loss=cross_entropy
      error=self.accuracy
    else:
      raise Exception('f_out not supported')

    f = conv_str_func(f)
    f_out = conv_str_func(f_out)

    Nl = len(Nodes)
    self.Nl = Nl # Number of hidden layers
    self.Nodes=Nodes #the array of the nodes

    self.f = [ ide ] + ( [f] * (Nl-2 )) + [ f_out ] #[f_in, f,f,f,f ,f_out] f[m](a[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = np.array( [None]*(Nl-1), dtype=object ) # matrici dei pesi 

    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)
    self.error = error

    self.train_history=[]
    self.valid_history=[]
    self.loss_history=[]

    # a[m+1] = f[m]( w[m]*a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
    for i in range(0,Nl-1): # Weight layer to layer, last column is bias
      self.w[i]=( 2*np.random.rand( Nodes[i+1], Nodes[i]+1 ) -1 )*np.sqrt(Nodes[i+1]+Nodes[i]) #the terms inside the sqrt are for xavier initialization

    # previous weights deltas tensor for momentum training
    self.deltas = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl-1) ] ,dtype=object) # prevous delta for momentum computation

  def forward_pass(self, u:np.ndarray ): 
    """
    compute activations and activation potentials
    """
    Nl = self.Nl
    v = [None]*Nl # potenziali attivazione v[m]
    a = [None]*Nl # attivazioni a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nodes[0],1): 
      u = u.reshape((self.Nodes[0],1))

    # compute activation and potentials for units in each layer
    v[0] = u
    a[0] = u # activation of input units is external input
    for m in range(self.Nl-1): 
      v[m+1] =  np.dot( self.w[m] , np.vstack((a[m],1)) ) # activation of bias units is always 1
      a[m+1] = self.f[m+1](v[m+1])
    return a,v

  def backward_pass(self, y, a, v): 
    """
    given activations and potentials compute error-propagation-coefficents
    v: activation potentials
    a: activation
    """
    Nl=self.Nl

    d = [None]*(self.Nl) # error-propagation-coefficents d[m]

    # reshape desired-output if needed
    if not y.shape == (self.Nodes[-1],1):
      y = y.reshape((self.Nodes[-1],1))

    # calculate error-propagation-coefficents for units in each layer
    d[Nl-1] = self.dl( y , a[Nl-1]) * self.df[Nl-1](v[Nl-1]) # error-propagation-coefficents of output units
    for m in range(Nl-2,-1,-1):
      d[m] =  np.dot(  np.delete( self.w[m].T , -1, 0)  , d[m+1]  ) * self.df[m](v[m])  # must get row (column) of bias weights out of the computation of propagation coefficents

    return d

  def compute_gradient(self,p): 
    """
    compute gradient of error over pattern p
    """
    Nl = self.Nl

    # pattern is composed of input and relative desired output
    x,y = p

    # compute activations and potentials
    a, v = self.forward_pass( x ) 

    # compute error-propagation-coefficents
    d = self.backward_pass( y, a, v ) 

    #compute gradient for each layer. To siumlate bias activation, i add to activations a 1 at the bottom
    grad = [ np.dot( d[m+1] , np.vstack( ( a[m], 1 ) ).T ) for m in range(Nl-1) ]

    return np.array(grad, dtype=object)

  def epoch(self, train_x:np.ndarray, train_y:np.ndarray, eta, a=1e-12,l=1e-12,bs=None):
    """
    Use all patterns in the given training set to execute an epoch of batch training with (possibly thickonov regularization)
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    a: momentum rate
    l: thickonov regularization rate
    bs: batch size
    """
    N = np.size(train_x,axis=0)
    if bs is None: bs = N # number of patterns in training set
    for _ in range(int(N/bs)):
      # compute gradient summing over partial gradients
      if bs!=N:
        i = np.random.randint(0,N,size=bs)
        p = sum( map( self.compute_gradient, zip( train_x[i],train_y[i] ) ) )/bs
      else:
        p = sum( map( self.compute_gradient, zip( train_x,train_y ) ) )/bs
  
      #compute deltas
      self.deltas = eta * p + a * self.deltas - l * self.w

      # update weights
      self.w += self.deltas



  def train(self, train_x, train_y, eta, a=1e-12, l=1e-12, bs=30, val_x=None, val_y=None, max_epochs=300, tresh=.01, mode='batch', shuffle_data=True, measure_interval=10, verbose=True):
    """
    Executes a maximum of max_epochs epochs of training using the function epoch_f in order to do regression of some function that maps input train_x->train_y.
    After each measure_interval epochs, mesures error on training set, and exits when training error falls below given treshold.
    Returns error at each mesurement calculated both on training and validation set, so you can plot them.
    Could use some early stopping mechanism through validation error.
    """
    
    # execute epochs of training until training is complete or max_epochs epochs are executed (or training error diverges)
    for i in range(max_epochs):

      # shuffle training set if needed
      """i would do this before the training"""
      if shuffle_data==True:
        train_x, train_y = shuffle(train_x, train_y)
      
      # execute an epoch of training
      self.epoch(train_x, train_y, eta, a=a, l=l ,bs=bs) # epoca di allenamento
      
      # after each measure_interval epochs of training do calculation:
      # decide if training is done and mesure training and validation error
    
      # mesure error on validation set if validation set is provided
      if val_x is not None:
        v = self.error(val_x,val_y) # Mean Squared Error on training set
        self.valid_history.append(v)

      # measure error on training set to decide if training is complete
      e = self.error(train_x,train_y) # error on training set
      self.train_history.append(e)

      # mesure loss on training set to get statistics
      loss = self.l(self.__call__(train_x).reshape(np.shape(train_y)),train_y)

      self.loss_history.append(loss)

      if verbose: 
        print(f'training error atm: {e}, validation error {v}, epoch={i}') 

        clear_output(wait=True)

      if i % measure_interval == 0:
        idx_m = int(i/measure_interval) # number of mesurements done 
      
        # if training is complete exit the loop. training is complete when training error falls below treshold tresh or 
        # error on training set is getting worse due to bad training parameters
        #i have put five to avoid errors and btw no net converges in less than 5 epochs
        if idx_m>5 and ( np.abs(e - self.train_history[-2])  < tresh or e > self.train_history[-2]):  # we quit training when error on training set falls below treshold
          if verbose: 
            print(f'final error: {e}')
          break

  def supply(self, u, categorical=False):
    """
    Supply an input to this network. The network computes its internal state and otuput of the network is activation of the last layer's units.
    u: input pattern
    returns output of the network given the supplied pattern
    """
    # reshape input if needed
    if not u.shape == (self.Nodes[0],1):
      u = u.reshape((self.Nodes[0],1))

    # calculate activation of units in each layer
    for m in range(self.Nl-1):
      u = self.f[m+1]( np.dot( self.w[m] , np.vstack((u,1)) ) )

    #return the output  
    if categorical==True: return smax_to_categorical(u)
    return np.copy(u)

  def __call__(self,U):
    """
    given sequence of input patterns, computes sequence of relative network's outputs.
    complied version of 
      return [float(self.predict(u)) for u in tx]
    U: sequence of input patterns.
    """
    # calculate sequence of outputs of the network when provided when given sequence of inputs
    return np.array(list( map( self.supply, U ) ))
  
  def test_error(self, X, Y):
    outs = self.__call__(X)
    return self.error(outs, Y.reshape(outs.shape))

  def accuracy(self, X, Y): #only valid for classification algorithms
    correct=0
    total=len(X)
    for x,y in zip (X,Y):
      # cannot do just self.supply(x,True)==y bc python compares element by element. must use np.equal(x,y) or .all()
      if np.array_equal(self.supply(x,True).reshape(-1), y.reshape(-1)): correct+=1
    return correct/total

  def MSE(self,X,Y):
    outs = self.__call__( X ) # actual outputs of the network on training set
    if outs.shape != Y.shape:
      outs = outs.reshape(Y.shape) # reshape when neede or error calculation doesn't work
    assert outs.shape == Y.shape
    return np.mean( np.square( outs - Y ) )

  def MED(self,X,Y):
    outs = self.__call__(X)
    total=0
    if outs.shape != Y.shape:
      outs = outs.reshape(Y.shape) # reshape when neede or error calculation doesn't work  
    delta=outs-Y  
    for d in delta:
      total+=np.linalg.norm(d)
    return total/len(delta)

  def save(self, filename):
    pickle_out=open(filename,'wb')
    pickle.dump(self,pickle_out)
    pickle_out.close()