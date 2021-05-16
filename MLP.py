#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

# general imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from IPython.display import clear_output

# activation functions
from numpy import tanh
from scipy.special import softmax
ide = lambda x : np.copy(x)
relu = lambda x: x*(x > 0)
#softmax = lambda x: np.exp(x - logsumexp(x, keepdims=True)) # implementazione scipy special

def to_categorical(y, num_classes=None, dtype='float32'): # code from keras implementation: keras.utils.to_categorical
  """Converts a class vector (integers) to binary class matrix.
  E.g. for use with categorical_crossentropy.
  Args:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.
  Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.
  """
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def smax_to_categorical(y):
  return to_categorical(np.argmax(y),len(y),dtype='int32').reshape(-1,1,1) #non torna int 

# loss functions:
squared_error = lambda y,d:  np.linalg.norm(y - d) ** 2 # categorical cross-entropy
cross_entropy = lambda y,d: -np.sum( d * np.log( y + np.finfo(float).eps ) )
MSE = lambda x,y: np.mean( np.square( x-y ) )

def derivative(f):
  """
  When f is an activation function, returns derivative w.r.t. potential of activation
  When f is a loss, returns derivative w.r.t. activation
  When f is cross_entropy and activation of output units is softmax, maths say derivative of loss w.r.t potential is one returned
  """
  if f == tanh:
    return lambda x: 1.0 - tanh(x)**2
  elif f == relu:
    return lambda x: 1*(x>=0)
  elif f == ide or f == softmax:
    return lambda x : x-x+1 
  elif f == squared_error or f == cross_entropy: 
    return lambda d,y: d-y


def conv_str_func(f):
  """
  if given a string of a function it gives you the actual function.
  if given the function it gives the string
  es: conv_str_func('relu')==relu()
  es: conv_str_func(relu)=='relu'
  """
  functions=[ relu , tanh , ide , softmax , squared_error , cross_entropy , MSE]
  strings = ['relu','tanh','ide','softmax','squared_error','cross_entropy','MSE']
  for i in range(len(functions)):
    if f==functions[i]: return strings[i]
    if f==strings[i]: return functions[i]
  raise Exception('Input not valid')


class MLP():

  def __init__(self, Nh=[10], Nu=1, Ny=1, f='tanh', f_out='ide' , w_range=.7):
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
      error=MSE
    elif f_out=='softmax':
      loss=cross_entropy
      error=self.accuracy
    else:
      raise Exception('f_out not supported')

    f = conv_str_func(f)
    f_out = conv_str_func(f_out)

    Nl = len(Nh)
    self.Nl = Nl # Number of layers
    self.Nu = Nu # Input unit
    self.Ny = Ny # Output unit
    self.Nh = Nh # Internal Unit

    self.f = [ ide ] + ( [f] * Nl ) + [ f_out ] #[f_in, f,f,f,f ,f_out] f[m](a[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = np.array( [None]*(Nl+1), dtype=object ) # matrici dei pesi 

    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)
    self.error = error

    self.train_history=[]
    self.valid_history=[]

    # a[m+1] = f[m]( w[m]*a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
    self.w[0] = ( 2*np.random.rand( Nh[0], Nu+1 ) -1 )*w_range# pesi input-to-primo-layer, ultima colonna e' bias. w[i,j] in [-1,1]
    for i in range(1, Nl):
      self.w[i] = ( 2*np.random.rand( Nh[i], Nh[i-1] + 1 )-1 )*w_range# pesi layer-to-layer, ultima colonna e' bias
    self.w[Nl] = ( 2*np.random.rand( Ny, Nh[Nl-1] + 1) -1 )*w_range# pesi ultimo-layer-to-output, ultima colonna e' bias
    
    # previous weights deltas tensor for momentum training
    self.deltas = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) # prevous delta for momentum computation

  def forward_pass(self, u:np.ndarray ): 
    """
    compute activations and activation potentials
    """
    Nl = self.Nl
    v = [None]*(Nl+2) # potenziali attivazione v[m]
    a = [None]*(Nl+2) # attivazioni a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))

    # compute activation and potentials for units in each layer
    v[0] = u
    a[0] = u # activation of input units is external input
    for m in range(self.Nl+1): 
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

    d = [None]*(self.Nl+2) # error-propagation-coefficents d[m]

    # reshape desired-output if needed
    if not y.shape == (self.Ny,1):
      y = y.reshape((self.Ny,1))

    # calculate error-propagation-coefficents for units in each layer
    d[Nl+1] = self.dl( y , a[Nl+1]) * self.df[Nl+1](v[Nl+1]) # error-propagation-coefficents of output units
    for m in range(Nl,-1,-1):
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
    grad = [ np.dot( d[m+1] , np.vstack( ( a[m], 1 ) ).T ) for m in range(Nl+1) ]

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
      if i % measure_interval == 0:
        idx_m = int(i/measure_interval) # number of mesurements done 

        # mesure error on validation set if validation set is provided
        if val_x is not None:
          outs_v = self.__call__( val_x ) # actual outputs of the network on validation set
          if outs_v.shape != val_y.shape:
            outs_v = outs_v.reshape(val_y.shape) # reshape when needed or error calculation doesn't work
          assert outs_v.shape == val_y.shape
          v = self.error(outs_v,val_y) # Mean Squared Error on training set
          self.valid_history.append(v)

        # measure error on training set to decide if training is complete
        outs_t = self.__call__( train_x ) # actual outputs of the network on training set
        if outs_t.shape != train_y.shape:
          outs_t = outs_t.reshape(train_y.shape) # reshape when neede or error calculation doesn't work
        assert outs_t.shape == train_y.shape

        e = self.error(outs_t,train_y) # error on training set
        self.train_history.append(e)

        if verbose: 
          print(f'training error atm: {e}') 

          clear_output(wait=True)

        
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
    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))

    # calculate activation of units in each layer
    for m in range(self.Nl+1):
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
      if self.supply(x,True)==y: correct+=1
    return correct/total
  
