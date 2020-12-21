#http://machinelearningmechanic.com/deep_learning/2019/09/04/cross-entropy-loss-derivative.html
#https://deepnotes.io/softmax-crossentropy

import numpy as np


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

  def forward_pass(self, u:np.ndarray ): # calcolo attivazioni e potenziali di attivazione
    if not u.shape == (self.Nu,1): 
      u = u.reshape((self.Nu,1))
    self.v[0] = u
    self.x[0] = u # attivazione untia' input e' l'input esterno
    for m in range(self.Nl+1): 
      self.v[m+1] =  np.dot( self.w[m] , np.vstack((self.x[m],1)) ) # attivazione untia' di bias sempre 1 #
      self.x[m+1] = self.f[m+1](self.v[m+1])

  def backward_pass(self, y ): # calcolo coefficenti di propagazione errore
    Nl=self.Nl
    if not y.shape == (self.Ny,1):
      y = y.reshape((self.Ny,1))
    self.d[Nl+1] = self.dl( y , self.x[Nl+1]) * self.df[Nl+1](self.v[Nl+1]) # coeff prop output
    for m in range(Nl,-1,-1):
      self.d[m] =  np.dot(  np.delete( self.w[m].T , -1, 0)  , self.d[m+1]  ) * self.df[m](self.v[m])  # devo levare la riga (colonna) dei bias qui 

  def compute_gradient(self): # date attivazioni e coefficenti errore calcolo gradiente
    Nl = self.Nl
    for m in range(Nl+1):
      self.grad[m] = np.dot( self.d[m+1] , np.vstack((self.x[m],1)).T )

  def online_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta):
    for i in range(np.size(train_x,axis=0)):
      self.forward_pass( train_x[i] ) # calcoli attivazione e potenziale
      self.backward_pass( train_y[i] ) # calcoli coeff di propagazionie
      self.compute_gradient() 

      for m in range(self.Nl+1):
        self.w[m] += eta * self.grad[m] 

  def batch_BP(self, train_x:np.ndarray, train_y:np.ndarray, eta, a=1e-12,l=1e-12):
    """
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

  def predict(self, u):
    if not u.shape == (self.Nu,1):
      u = u.reshape((self.Nu,1))
    
    self.x[0] = u
    for m in range(self.Nl+1):
      self.x[m+1] = self.f[m+1]( np.dot( self.w[m] , np.vstack((self.x[m],1)) ) )
    return np.copy(self.x[self.Nl+1])
      