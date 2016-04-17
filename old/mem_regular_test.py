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
future_steps = 5
memory_length = 10
 
def prepare_bbox():
	global n_features, n_actions, max_time
 
	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("../levels/test_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		max_time = bbox.get_max_time()  
 
def run_bbox():
  print '####### Testing #######'
  has_next = 1
  prepare_bbox()
  
  def model(X, w):
    #h = T.sum(T.batched_dot(w, X), axis=0)
    h = T.tensordot(w, X, [[2,0],[1,0]])
    py_x = T.nnet.softmax(h.T)
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
    
  with open('LearnedParams.bin', 'rb') as fp:
        params = cPickle.load(fp)
  w = params[0]  
    
  X = T.tensor3()
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
  t = time.time()

  states = np.zeros((memory_length, n_features + 1, 1))
  while has_next:
    state = bbox.get_state()
    t_state = np.reshape(np.append(state, [1]), (1, n_features+1, 1))
    s = np.delete(states, 0, axis=0)
    states = np.append(s, t_state, axis=0)
    step = bbox.get_time()
    if step % 10000 == 0:
      print '########################'
      print 'Iteration:       {:7d}'.format(step)
      print 'Time:               {:2.2f}'.format(time.time() - t)
      print 'Current Score:  {:6.2f}'.format(bbox.get_score())
      t = t = time.time()
    action = predict(states)
    for _ in range(future_steps):
      has_next = bbox.do_action(action)
  

  bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
	run_bbox()