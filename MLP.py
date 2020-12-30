#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

import numpy as np
import pandas as pd
from IPython.display import clear_output
from sklearn.utils import shuffle

# funzioni attivazione
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

# funzioni loss
squared_error = lambda y,d:  np.linalg.norm(y - d) ** 2 # categorical cross-entropy
cross_entropy = lambda y,d: -np.sum( d * np.log( y + np.finfo(float).eps ) )

MSE = lambda x,y: np.mean( np.square( x-y ) )


def derivative(f):
  if f == tanh:
    return lambda x: 1.0 - tanh(x)**2
  elif f == relu:
    return lambda x: 1*(x>=0)
  elif f == ide or f == softmax:
    return lambda x : x-x+1
  elif f == squared_error or f == cross_entropy:
    return lambda d,y: d-y


class MLP():

  def __init__(self, Nh=[10], Nu=1, Ny=1, f=tanh, f_out=ide , w_range=.7, w_scale=2, loss=squared_error, error=MSE):
    
    if loss == cross_entropy:
      assert f_out == softmax, 'if using cross-entropy loss, must use softmax as output activation function'

    Nl = len(Nh)
    self.Nl = Nl # numero layer
    self.Nu = Nu # unita' input
    self.Ny = Ny # unita' output
    self.Nh = Nh # unita' interne

    self.f = [ide] + ([f]*Nl) + [f_out] #[f_in, f,f,f,f ,f_out] f[m](a[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = np.array([None]*(Nl+1)) # matrici dei pesi 

    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)
    self.error = error

    # a[m+1] = f[m]( w[m]*a[m] ) a[m] = (Nh,1) a[m+1] = (Nh,1) w[m] = (Nh,Nh)
    self.w[0] = ( 2*np.random.rand( Nh[0], Nu+1 ) -1 )*w_scale # pesi input-to-primo-layer, ultima colonna e' bias. w[i,j] in [-1,1]
    for i in range(1, Nl):
      self.w[i] = ( 2*np.random.rand( Nh[i], Nh[i-1] + 1 )-1 )*w_scale # pesi layer-to-layer, ultima colonna e' bias
    self.w[Nl] = ( 2*np.random.rand( Ny, Nh[Nl-1] + 1) -1 )*w_scale # pesi ultimo-layer-to-output, ultima colonna e' bias

  def forward_pass(self, u:np.ndarray ): 
    """
    Calcolo attivazioni e potenziali di attivazione
    """
    Nl = self.Nl
    v = [None]*(Nl+2) # potenziali attivazione v[m]
    a = [None]*(Nl+2) # attivazioni a[m] = f[m](v[m])

    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))
    v[0] = u
    a[0] = u # attivazione untia' input e' l'input esterno
    for m in range(self.Nl+1): 
      v[m+1] =  np.dot( self.w[m] , np.vstack((a[m],1)) ) # attivazione untia' di bias sempre 1 #
      a[m+1] = self.f[m+1](v[m+1])
    return a,v

  def backward_pass(self, y, a, v): 
    """
    Dati attivazioni e potenziali calcolo coefficenti di propagazione errore
    """
    Nl=self.Nl
    d = [None]*(self.Nl+2) # coefficenti di propagazione d[m]
    if not y.shape == (self.Ny,1):
      y = y.reshape((self.Ny,1))
    d[Nl+1] = self.dl( y , a[Nl+1]) * self.df[Nl+1](v[Nl+1]) # coeff prop output
    for m in range(Nl,-1,-1):
      d[m] =  np.dot(  np.delete( self.w[m].T , -1, 0)  , d[m+1]  ) * self.df[m](v[m])  # devo levare la riga (colonna) dei bias qui 
    return d

  def compute_gradient(self,p): 
    """
    Date attivazioni, potenziali e coefficenti propagazione errore calcolo gradiente
    """
    Nl = self.Nl
    x,y = p
    a, v = self.forward_pass( x ) # calcoli attivazione e potenziale
    d = self.backward_pass( y, a, v ) # calcoli coeff di propagazionie
    grad = [ np.dot( d[m+1] , np.vstack( ( a[m], 1 ) ).T ) for m in range(Nl+1) ]
    return np.array(grad)

  def epoch_batch_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta, a=1e-12,l=1e-12):
    """
    Use all patterns in the given training set to execute an epoch of batch training
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    a: momentum rate
    l: thickonov regularization rate
    """
    old_deltas = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) # momento
    p = np.array( [ np.zeros(self.w[i].shape) for i in range(self.Nl+1) ] ,dtype=object) # somma parziale gradienti
    N = np.size(train_x,axis=0) # numero sample in training set

    comp_grad = self.compute_gradient
    p = sum( map( comp_grad, zip(train_x,train_y ) ) )/N

    deltas = eta * p + a * old_deltas - l * self.w
    self.w += deltas
    old_deltas = deltas
  
  def train(self, train_x, train_y, eta, a=1e-12, l=1e-12, val_x=None, val_y=None, max_epochs=300, tresh=.01, epoch_f=None, shuffle_data=True, measure_interval=10):
    """
    Executes a maximum of max_epochs epochs of training using the function epoch_f in order to do regression of some function that maps input train_x->train_y.
    After each measure_interval epochs, mesures error on training set, and exits when training error falls below given treshold.
    Returns error at each mesurement calculated both on training and validation set, so you can plot them.
    Could use some early stopping mechanism through validation error.
    """
    e = [None]*max_epochs 
    v = [None]*max_epochs
    for i in range(max_epochs):
      if shuffle_data==True:
        train_x, train_y = shuffle(train_x, train_y)
      
      epoch_f( train_x, train_y, eta, a=a, l=l ) # epoca di allenamento
      
      if i % measure_interval == 0:
        i=int(i/measure_interval)
        outs_t = self.supply_sequence( train_x ) # actual outputs of the network
        if outs_t.shape != train_y.shape:
          outs_t = outs_t.reshape(train_y.shape)
        assert outs_t.shape == train_y.shape
        e[i] = self.error(outs_t,train_y) # Mean Squared Error on training set
        if val_x is not None:
          outs_v = self.supply_sequence( val_x ) # actual outputs of the network
          v[i] = self.error(outs_v,val_y) # Mean Squared Error on training set
        if i>2 and ( e[i] < tresh or e[i]>e[i-1]):  # we quit training when error on training set falls below treshold
          print(f'final error: {e[i]}')
          break
        print(f'training error atm: {e[i]}') 
        clear_output(wait=True)
    return e, v

  def supply(self, u):
    """
    Supply an input to this network. The network computes its internal state and otuput of the network is activation of the last layer's units.
    u: input pattern
    returns output of the network given the supplied pattern
    """
    a = [None]*(self.Nl+2) # attivazioni a[m] = f[m](v[m])

    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))
    
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
    sup = self.supply
    return np.array(list( map( lambda u : sup(u) , U ) ))
  
  def test(self, X, Y):
    outs = self.supply_sequence(X)
    return self.error(outs, Y)
  
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
