import theano
from theano import tensor as T
import numpy as np
import interface as bbox
import cPickle
import time
cimport interface as cbbox

theano.config.floatX = 'float32' 
n_features = 36
n_actions = 4
max_time = -1
future_steps = 20
memory_length = 1

lambda1 = 0.00005
lambda2 = 0.0002 

cdef float cfs[4][37]
cdef float st[37]
  
cdef float fast_predict(float* state):
  cdef:
    float sum = 0
    int best_act = -1
    float best_val = -1e9
    float val

  for act in xrange(n_actions):
    sum = 0
    for i in xrange(37):
      sum = sum + cfs[act][i] * state[i]
    val = sum
    
    if val > best_val:
      best_val = val
      best_act = act
    
  return best_act

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
  coefs = np.loadtxt(filename).reshape(1, n_actions, n_features + 1)
  coefs = np.asarray(coefs,  dtype=theano.config.floatX)

def run_bbox():
  has_next = 1

  prepare_bbox()
  load_regression_coefs("best_overwrite.txt")
  
  def get_states(states):
    state = bbox.get_state()
    t_state = np.reshape(np.append(state, [1]), (1, n_features+1, 1))
    s = np.delete(states, 0, axis=0)
    states = np.append(s, t_state, axis=0)  
    return states

  def calc_best_action_using_checkpoint(w, states):
    action_scores = []
    checkpoint_id = bbox.create_checkpoint()
    
    new_w = np.reshape(w.get_value(), (4, 37))
    
    for i in xrange(4):
      for j in xrange(37):
        cfs[i][j] = new_w[i][j]
    
    for first_action in range(n_actions):
      bbox.do_action(first_action)
      for _ in range(future_steps):
        states = get_states(states)
        
        for i in xrange(37):
          st[i] = states[0][i][0]
        action = fast_predict(st)
        bbox.do_action(action)
      
      action_scores.append(bbox.get_score())
      bbox.load_from_checkpoint(checkpoint_id)
    
    result = np.zeros(n_actions)
    result[np.argmax(action_scores)] = 1
    return result
  
  def model(X, w):
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
  X = T.tensor3()
  # w = theano.shared(np.zeros((memory_length, n_actions, n_features + 1), dtype=theano.config.floatX))
  w = theano.shared(coefs)
  py_x = model(X, w)
  a = T.argmax(py_x)
  
  Y = T.vector()
  params = [w]
  
  # l1 = sum([T.sum(abs(p)) for p in params])
  # l2 = sum([T.sum(T.pow(p, 2)) for p in params])
  cost = T.mean(T.nnet.categorical_crossentropy(py_x, T.nnet.softmax(Y))) 
  updates = RMSprop(cost, params)
  
  train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
  predict = theano.function(inputs=[X], outputs=a, allow_input_downcast=True)
  t_state = np.ones(37) 
  states = np.zeros((memory_length, n_features + 1, 1))  
  start = time.time()  
  print 'Done.' 
  
  while has_next:
    states = get_states(states)
    action_scores = calc_best_action_using_checkpoint(w, states)
    cost = train(states, action_scores)
    action = predict(states)
    for _ in range(1):
      has_next = bbox.do_action(action)
      
    t = bbox.get_time()
    if t % 10000 == 0:
      print t, ':   score =', bbox.get_score(), ', time =', time.time() - start
      start = time.time()  
  
  with open('LearnedParams.bin','wb') as fp:
    cPickle.dump(params,fp)
  bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
  run_bbox()