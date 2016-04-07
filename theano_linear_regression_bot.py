import theano
from theano import tensor as T
import numpy as np
import interface as bbox
import cPickle 


n_features = 36
n_actions = 4
max_time = -1

 
def prepare_bbox():
	global n_features, n_actions, max_time
 
	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("../levels/train_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		max_time = bbox.get_max_time()
 
 
def load_regression_coefs(filename):
	global coefs
	coefs = np.loadtxt(filename).reshape(n_actions, n_features + 1)

 
def run_bbox():
  has_next = 1
	
  prepare_bbox()
  load_regression_coefs("reg_coefs.txt")
  
  def model(X, w):
    return T.dot(w, X)

  X = T.vector()
  w = theano.shared(np.asarray(coefs, dtype=theano.config.floatX))
  a = T.argmax(model(X, w))
  predict = theano.function(inputs=[X], outputs=a, allow_input_downcast=True)
  t_state = np.ones(37)
  while has_next:
    state = bbox.get_state()
    t_state = np.append(state, [1])
    #state = np.asarray(state, dtype=theano.config.floatX)
    action = predict(t_state)
    has_next = bbox.do_action(action)
 
  bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
	run_bbox()