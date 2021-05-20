import numpy as np

#softmax = lambda x: np.exp(x - logsumexp(x, keepdims=True)) # implementazione scipy special


#some functions
def ide(x):return np.copy(x)
def relu(x):return x*(x>0)

# loss functions:
def squared_error(y,d): return np.linalg.norm(y - d) ** 2 
# categorical cross-entropy
def cross_entropy (y,d): return -np.sum( d * np.log( y + np.finfo(float).eps ) )
#MSE = lambda x,y: np.mean( np.square( x-y ) )

#some derivatives
def dtanh(x): return 1.0 - np.tanh(x)**2
def drelu(x): return 1*(x>0)
def dide (x): return x-x+1
def dloss(d,y): return d-y



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
