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