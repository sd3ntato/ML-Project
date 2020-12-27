#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

import numpy as np
from IPython.display import clear_output
from sklearn.utils import shuffle

# funzioni attivazione
from numpy import tanh
ide = lambda x : np.copy(x)
relu = lambda x: x*(x > 0)

# funzioni loss
squared_error = lambda y,d:  np.linalg.norm(y - d) ** 2

MSE = lambda x,y: np.mean( np.square( x-y ) )


def derivative(f):
  if f == tanh:
    return lambda x: 1.0 - np.tanh(x)**2
  elif f == relu:
    return lambda x: 1*(x>=0)
  elif f == ide:
    return lambda x : x-x+1
  elif f == squared_error:
    return lambda d,y: d-y


class MLP():

  def __init__(self, Nh=16, Nl=2, Nu=1, Ny=1, f=tanh, f_out=ide ,w_scale=.05, loss=squared_error):

    self.f = [ide] + ([f]*Nl) + [f_out] #[f_in, f,f,f,f ,f_out] f[m](x[m])
    self.df = [ derivative(f) for f in self.f] # df[m](v[m])
    self.w = [None]*(Nl+1) # matrici dei pesi 
    self.v = [None]*(Nl+2) # potenziali attivazione v[m]
    self.x = [None]*(Nl+2) # attivazioni x[m] = f[m](v[m])
    self.d = [None]*(Nl+2) # coefficenti di propagazione d[m]
    self.grad = [None]*(Nl+1) # gradienti w[m] = w[m] + lr*grad[m]
    self.l = loss # funzione loss (y-d)**2
    self.dl = derivative(loss) # (y-d)

    self.Nl = Nl # numero layer
    self.Nu = Nu # unita' input
    self.Ny = Ny # unita' output
    self.Nh = Nh # unita' interne

    # x[m+1] = f[m]( w[m]*x[m] ) x[m] = (Nh,1) x[m+1] = (Nh,1) w[m] = (Nh,Nh)

    self.w[0] = (2*np.random.rand(Nh,Nu+1)-1)*w_scale # pesi input-to-primo-layer, ultima colonna e' bias. w[i,j] in [-1,1]
    for i in range(1,Nl):
      self.w[i] = (2*np.random.rand(Nh,Nh+1)-1)*w_scale # pesi layer-to-layer, ultima colonna e' bias
    self.w[Nl] = (2*np.random.rand(Ny,Nh+1)-1)*w_scale # pesi ultimo-layer-to-output, ultima colonna e' bias

  def forward_pass(self, u:np.ndarray ): 
    """
    Calcolo attivazioni e potenziali di attivazione
    """
    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))
    self.v[0] = u
    self.x[0] = u # attivazione untia' input e' l'input esterno
    for m in range(self.Nl+1): 
      self.v[m+1] =  np.dot( self.w[m] , np.vstack((self.x[m],1)) ) # attivazione untia' di bias sempre 1 #
      self.x[m+1] = self.f[m+1](self.v[m+1])

  def backward_pass(self, y ): 
    """
    Calcolo coefficenti di propagazione errore
    """
    Nl=self.Nl
    if not y.shape == (self.Ny,1):
      y = y.reshape((self.Ny,1))
    self.d[Nl+1] = self.dl( y , self.x[Nl+1]) * self.df[Nl+1](self.v[Nl+1]) # coeff prop output
    for m in range(Nl,-1,-1):
      self.d[m] =  np.dot(  np.delete( self.w[m].T , -1, 0)  , self.d[m+1]  ) * self.df[m](self.v[m])  # devo levare la riga (colonna) dei bias qui 

  def compute_gradient(self): 
    """
    Date attivazioni e coefficenti errore calcolo gradiente
    """
    Nl = self.Nl
    for m in range(Nl+1):
      self.grad[m] = np.dot( self.d[m+1] , np.vstack((self.x[m],1)).T )

  def epoch_online_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta):
    """
    Use all patterns in the given training set to execute an epoch of online training
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    """
    for i in range(np.size(train_x,axis=0)):
      self.forward_pass( train_x[i] ) # calcoli attivazione e potenziale
      self.backward_pass( train_y[i] ) # calcoli coeff di propagazionie
      self.compute_gradient() 

      for m in range(self.Nl+1):
        self.w[m] += eta * self.grad[m] 

  def epoch_batch_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta, a=1e-12,l=1e-12):
    """
    Use all patterns in the given training set to execute an epoch of batch training
    train_x : input in training set
    train_y : output in training set
    eta: learning rate
    a: momentum rate
    l: thickonov regularization rate
    """
    old_deltas = [np.zeros(self.w[i].shape) for i in range(self.Nl+1)] # momento
    p = [np.zeros(self.w[i].shape) for i in range(self.Nl+1)] # somma parziale gradienti
    N = np.size(train_x,axis=0) # numero sample in training set
    for i in range(N): # per ogni pattern
      self.forward_pass( train_x[i] ) 
      self.backward_pass( train_y[i] )
      self.compute_gradient() # gradiente al passo i

      for m in range(self.Nl+1):
        p[m] += self.grad[m] * (1/N)

    for m in range(self.Nl+1):
      self.w[m] += eta * p[m] + a * old_deltas[m] - l * self.w[m]
      old_deltas[m] = eta * p[m] + a * old_deltas[m] - l * self.w[m]

  def supply(self, u):
    """
    Supply an input to this network. The network computes its internal state and otuput of the network is activation of the last layer's units.
    u: input pattern
    returns output of the network given the supplied pattern
    """
    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))
    
    self.x[0] = u
    for m in range(self.Nl+1):
      self.x[m+1] = self.f[m+1]( np.dot( self.w[m] , np.vstack((self.x[m],1)) ) )
    return np.copy(self.x[self.Nl+1])

  def supply_sequence(self,U):
    """
    given sequence of input patterns, computes sequence of relative network's outputs.
    complied version of 
      return [float(self.predict(u)) for u in tx]
    U: sequence of input patterns.
    """
    sup = self.supply
    return list( map( lambda u :float( sup(u) ), U ) )
  
  def classify(self,u):
    """
    classify sample. To be written properly
    u: imput pattern
    """
    return -1 if float( self.supply(u) )<0 else 1

  def regression_train(self, train_x, train_y, eta, a=1e-12, l=1e-12, val_x=None, val_y=None, max_epochs=300, tresh=.01, epoch_f=None, shuffle_data=True, measure_interval=10):
    """
    Executes a maximum of max_epochs epochs of training using the function epoch_f.
    After measure_interval epochs, mesures error on training set, and exits when training error falls below given treshold.
    Returns error at each mesurement calculated both on training and validation set, so you can plot them.
    Could use some early stopping mechanism through validation error.
    """
    e = [None]*max_epochs
    v = [None]*max_epochs
    for i in range(max_epochs):
      if shuffle_data==True:
        train_x, train_y = shuffle(train_x, train_y)
      
      epoch_f( train_x, train_y, eta, a=a, l=l )
      
      if i % measure_interval == 0:
        i=int(i/measure_interval)
        outs_t = self.supply_sequence( train_x ) # actual outputs of the network
        e[i] = MSE(outs_t,train_y) # Mean Squared Error on training set
        if val_x is not None:
          outs_v = self.supply_sequence( val_x ) # actual outputs of the network
          v[i] = MSE(outs_v,val_y) # Mean Squared Error on training set
        if i>2 and ( e[i] < tresh or e[i]>e[i-1]):  # we quit training when error on training set falls below treshold
          print(f'final error: {e[i]}')
          break
        print(f'training error atm: {e[i]}') 
        clear_output(wait=True)
    return e, v

def score(o,d): 
  """
  percentage score of nework in binary classification task. To be written properly
  o: sequence of ouputs of the network, 
  d: sequence of desired outputs
  """
  return ( o - d == 0).sum()/len(o)*100

from itertools import product  

def combinations(Nh,Nls):
  res = []
  for Nl in Nls :
    res.extend(list(product(Nh,repeat=Nl)))
  return res

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