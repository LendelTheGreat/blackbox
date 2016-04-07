import theano
from theano import tensor as T
import numpy as np
import interface as bbox
import cPickle 

theano.config.floatX = 'float32' 
n_features = 36
n_actions = 4
max_time = -1
future_steps = 100
 
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

def calc_best_action_using_checkpoint():
  checkpoint_id = bbox.create_checkpoint()
  action_scores = []

  for action in range(n_actions):
    for _ in range(future_steps):
      bbox.do_action(action)
		
    action_scores.append(bbox.get_score())

    bbox.load_from_checkpoint(checkpoint_id)

  return action_scores

  
 
def run_bbox():
  has_next = 1
	
  prepare_bbox()
  load_regression_coefs("best_overwrite.txt")
  
  def model(X, w):
    py_x = T.nnet.softmax(T.dot(w, X))
    return py_x
  
  def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
      acc = theano.shared(p.get_value() * 0.)
      acc_new = rho * acc + (1 - rho) * g ** 2
      gradient_scaling = T.sqrt(acc_new + epsilon)
      g = g / gradient_scaling
      updates.append((acc, acc_new))
      updates.append((p, p - lr * g))
    return updates
    
  print 'Compiling Theano graph...'
  X = T.vector()
  w = theano.shared(np.asarray(coefs, dtype=theano.config.floatX))
  py_x = model(X, w)
  a = T.argmax(py_x)
  
  Y = T.vector()
  params = [w]
  cost = T.mean(T.nnet.categorical_crossentropy(py_x, T.nnet.softmax(Y))) 
  updates = RMSprop(cost, params)
  
  train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
  predict = theano.function(inputs=[X], outputs=a, allow_input_downcast=True)
  t_state = np.ones(37) 
  
  
  print 'Done.' 

  while has_next:
    state = bbox.get_state()
    t_state = np.append(state, [1])
    time = bbox.get_time()
    if time % 10000 == 0:
      print time
    action_scores = calc_best_action_using_checkpoint()
    cost = train(t_state, action_scores)
    action = predict(t_state)
    for _ in range(1):
      has_next = bbox.do_action(action)
  
  with open('LearnedParams.bin','wb') as fp:
    cPickle.dump(params,fp)
  bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
	run_bbox()