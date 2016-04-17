import theano
from theano import tensor as T
import numpy as np
import interface as bbox
import cPickle 
import time

theano.config.floatX = 'float32' 
n_features = 36
n_actions = 4
max_time = -1

def prepare_bbox(training_set='train'):
  """
  training_set is either 'test' or 'train'
  """ 
  global n_features, n_actions
  bbox.load_level('../levels/{}_level.data'.format(training_set), verbose=0)
  n_features = bbox.get_num_of_features()
  n_actions = bbox.get_num_of_actions()
  
 
def load_regression_coefs(filename):
	global coefs
	coefs = np.loadtxt(filename).reshape(n_actions, n_features + 1)

def load_squared_coefs(filename):
  n_actions = 4
  n_features = 36
  
  coefs = np.loadtxt(filename).reshape(n_actions, 2*n_features + 1)
  free_coefs = coefs[:,-1]
  coefs = coefs[:,:-1]
  
  c = np.zeros((n_actions,n_features))
  cc = np.zeros((n_actions,n_features))
  fc = np.zeros((1,n_actions))
  
  for i in xrange(n_actions):
    for j in xrange(2*n_features):
      if j % 2 == 0:
        c[i][j/2] = coefs[i][j]
      else:
        cc[i][(j-1)/2] = coefs[i][j]
    fc[0,i] = free_coefs[i]
  return c,cc,fc
 
def run_bbox():
  has_next = 1
  prepare_bbox()
  c,cc,fc = load_squared_coefs("coefs_squared_test_after10.txt")
  max_time = bbox.get_max_time()
  print 'Compiling Theano graph...'
  def model(X, w, w2, b):
    sum = T.dot(w, X) + T.dot(w2, X*X) + b
    predicted_action = T.argmax(sum)
    return predicted_action
  X = T.vector()
  w = theano.shared(np.asarray(c, dtype=theano.config.floatX))
  w2 = theano.shared(np.asarray(cc, dtype=theano.config.floatX))
  b = theano.shared(np.asarray(fc, dtype=theano.config.floatX))
  predicted_action = model(X, w, w2, b)
  params = [w, w2, b] 
  predict = theano.function(inputs=[X], outputs=predicted_action, allow_input_downcast=True)
  print 'Done. Progress:' 
  start = time.time()

  while has_next:
    state = bbox.get_state()
    t = bbox.get_time()
    if t % 100000 == 0:
      print '{0:.2f}%'.format(100.0 * t/max_time)
    action = predict(state)
    has_next = bbox.do_action(action)
  print 'Took {0:.2f} s'.format(time.time() - start)
  bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
	run_bbox()