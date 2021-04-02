#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

# general imports
import numpy as np
import pandas as pd
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

def get_f(f):
  """
  activation function: string->function
  """ 
  if f=='relu':
    return relu
  elif f=='tanh':
    return tanh

def get_f_out(f):
  """
  activation function of output units: string->function
  """ 
  if f=='ide':
    return ide
  elif f=='softmax':
    return softmax

def get_loss(f):
  """
  loss function: string->function
  """ 
  if f=='squared_error':
    return squared_error
  elif f=='cross_entropy':
    return cross_entropy

def get_error(f):
  """
  error function: string->function
  """ 
  if f=='MSE':
    return MSE
  elif f=='cross_entropy':
    return cross_entropy

class MLP():

  def __init__(self, Nh=[10], Nu=1, Ny=1, f='tanh', f_out='ide' , w_range=.7, w_scale=2, loss='squared_error', error='MSE'):
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
      
    if loss == 'cross_entropy':
      assert f_out == 'softmax', 'if using cross-entropy loss, must use softmax as output activation function'

    f = get_f(f)
    f_out = get_f_out(f_out)
    loss = get_loss(loss)
    error = get_error(error)

    Nl = len(Nh)
    self.Nl = Nl # numero layer
    self.Nu = Nu # unita' input
    self.Ny = Ny # unita' output
    self.Nh = Nh # unita' interne

    self.f = [ ide ] + ( [f] * Nl ) + [ f_out ] #[f_in, f,f,f,f ,f_out] f[m](a[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = np.array( [None]*(Nl+1), dtype=object ) # matrici dei pesi 

    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)
    self.error = error

    # a[m+1] = f[m]( w[m]*a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
    self.w[0] = np.round( ( 2*np.random.rand( Nh[0], Nu+1 ) -1 )*w_range, w_scale )# pesi input-to-primo-layer, ultima colonna e' bias. w[i,j] in [-1,1]
    for i in range(1, Nl):
      self.w[i] = np.round( ( 2*np.random.rand( Nh[i], Nh[i-1] + 1 )-1 )*w_range, w_scale )# pesi layer-to-layer, ultima colonna e' bias
    self.w[Nl] = np.round( ( 2*np.random.rand( Ny, Nh[Nl-1] + 1) -1 )*w_range, w_scale )# pesi ultimo-layer-to-output, ultima colonna e' bias

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

    return np.array(grad)

  def epoch_batch_BP(self, old_deltas, train_x:np.ndarray, train_y:np.ndarray, eta, a=1e-12,l=1e-12):
    """
    Use all patterns in the given training set to execute an epoch of batch training with (possibly thickonov regularization)
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    a: momentum rate
    l: thickonov regularization rate
    """
    N = np.size(train_x,axis=0) # number of patterns in training set

    # compute gradient summing over partial gradients
    comp_grad = self.compute_gradient
    p = sum( map( comp_grad, zip( train_x,train_y ) ) )/N

    #compute deltas
    deltas = eta * p + a * old_deltas - l * self.w

    # update weights and old_deltas values
    self.w += deltas
    
    # current change returned to be saved and then passed again to this function as old_deltas
    return deltas
  
  def train(self, train_x, train_y, eta, a=1e-12, l=1e-12, val_x=None, val_y=None, max_epochs=300, tresh=.01, mode='batch', shuffle_data=True, measure_interval=10, verbose=True):
    """
    Executes a maximum of max_epochs epochs of training using the function epoch_f in order to do regression of some function that maps input train_x->train_y.
    After each measure_interval epochs, mesures error on training set, and exits when training error falls below given treshold.
    Returns error at each mesurement calculated both on training and validation set, so you can plot them.
    Could use some early stopping mechanism through validation error.
    """
    e = [None]*max_epochs  # array of training errors for each epoch
    v = [None]*max_epochs # array of validation errors for each epoch
    
    # set function that does training epoch
    epoch_f=None 
    if mode=='batch':
      epoch_f = self.epoch_batch_BP
    
    # previous weights deltas tesnsor for momentum training
    old_deltas = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) # prevous delta for momentum computation

    # execute epochs of training until training is complete or max_epochs epochs are executed (or training error diverges)
    for i in range(max_epochs):

      # shuffle training set if needed
      if shuffle_data==True:
        train_x, train_y = shuffle(train_x, train_y)
      
      # execute an epoch of training
      old_deltas = epoch_f( old_deltas, train_x, train_y, eta, a=a, l=l ) # epoca di allenamento
      
      # after each measure_interval epochs of training do calculation:
      # decide if training is done and mesure training and validation error
      if i % measure_interval == 0:
        idx_m = int(i/measure_interval) # number of mesurements done 

        # mesure error on validation set if validation set is provided
        if val_x is not None:
          outs_v = self.supply_sequence( val_x ) # actual outputs of the network on validation set
          if outs_v.shape != val_y.shape:
            outs_v = outs_v.reshape(val_y.shape) # reshape when needed or error calculation doesn't work
          assert outs_v.shape == val_y.shape
          v[idx_m] = self.error(outs_v,val_y) # Mean Squared Error on training set

        # measure error on training set to decide if training is complete
        outs_t = self.supply_sequence( train_x ) # actual outputs of the network on training set
        if outs_t.shape != train_y.shape:
          outs_t = outs_t.reshape(train_y.shape) # reshape when neede or error calculation doesn't work
        assert outs_t.shape == train_y.shape
        e[idx_m] = self.error(outs_t,train_y) # error on training set
        if verbose: 
          print(f'training error atm: {e[idx_m]}') 
          clear_output(wait=True)

        
        # if training is complete exit the loop. training is complete when training error falls below treshold tresh or 
        # error on training set is getting worse due to bad training parameters
        if i>0 and ( e[idx_m] < tresh or e[idx_m] > e[idx_m-1]):  # we quit training when error on training set falls below treshold
          if verbose: 
            print(f'final error: {e[idx_m]}')
          break
        # clear_output(wait=True)

    return e, v

  def supply(self, u):
    """
    Supply an input to this network. The network computes its internal state and otuput of the network is activation of the last layer's units.
    u: input pattern
    returns output of the network given the supplied pattern
    """
    a = [None]*(self.Nl+2) # attivazioni a[m] = f[m](v[m])

    # reshape input if needed
    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))
    
    # calculate activation of units in each layer
    a[0] = u
    for m in range(self.Nl+1):
      a[m+1] = self.f[m+1]( np.dot( self.w[m] , np.vstack((a[m],1)) ) )
    return np.copy(a[self.Nl+1])

  def supply_sequence(self,U):
    """
    given sequence of input patterns, computes sequence of relative network's outputs.
    complied version of 
      return [float(self.predict(u)) for u in tx]
    U: sequence of input patterns.
    """
    # calculate sequence of outputs of the network when provided when given sequence of inputs
    sup = self.supply
    return np.array(list( map( lambda u : sup(u) , U ) ))
  
  def test_error(self, X, Y):
    outs = self.supply_sequence(X)
    return self.error(outs, Y.reshape(outs.shape))
  
  """
  alla fine questa non e' che serva davvero.....
  def classify(self,u):
    classify sample. To be written properly
    u: imput pattern
    if self.f_out == softmax:
      return -1 if float( self.supply(u) )<0 else 1
  """

"""
  questa andrebbe riscritta bellina come quella batch
  def epoch_online_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta):
    ""
    Use all patterns in the given training set to execute an epoch of online training
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    ""
    for i in range(np.size(train_x,axis=0)):
      
      self.compute_gradient() 

      for m in range(self.Nl+1):
        self.w[m] += eta * self.grad[m] 
"""

"""
def score(o,d): 
  percentage score of nework in binary classification task. To be written properly
  o: sequence of ouputs of the network, 
  d: sequence of desired outputs
  return ( o - d == 0).sum()/len(o)*100
"""

"""
from itertools import product  
def combinations(Nh,Nls):
    x=[]
    for Nl in Nls :
        x.append(list(permutations(Nh_monk,Nls_monk)))
    
    combinedList=[*x[0],*x[1],*x[2]]
    return combinedList

Nh = [2, 3, 4]
Nls = [2, 3]

grid = {
  'units' : combinations(Nh,Nls),
  'lr' : [1e-01, 1e-02, 1e-03],  # learning rate
  'a' : [1e-01, 1e-02, 1e-03, 0],  # momento
  'l' : [1e-01, 1e-02, 1e-03, 1e-10, 1e-12, 0],  # regolarizzazione
  'f' : ['relu','tanh'], # funzione attivazione
  'w_init' : [0.7, 1e-02], # scalining iniziale pesi
}

def k_fold_CV(train_x, train_y, k, n_init, grid):
  # splittare il training set in k fold
  splits = split( train_x, train_y)  

  # per ogni configurazione in grid:
      # per ogni split in splits:
        # inizializza n_init reti con configurazione
        # allena
        # testa
        # salva la migliore configurazione

"""
