import interface as bbox
import numpy as np
import time
from random import *
import cPickle

ac = np.zeros((4, 4))
cc = np.zeros((4, 36))
c = np.zeros((4, 36))
fc = np.zeros(4)

step = 6000
score_threshold = 5
grad_update = 0.725
ac_gradient = 0.2
ac_grads = np.zeros((4, 4)) + ac_gradient

last_actions = np.zeros(4)

def get_action_by_state_fast(state):
  best_act = -1
  best_val = -1e9

  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state)
    if val > best_val:
      best_val = val
      best_act = act
 
  for act in xrange(n_actions):
    last_actions[act] = 0
  last_actions[best_act] = 1
 
  return best_act

n_features = 36
n_actions = 4

def prepare_bbox():
    global n_features, n_actions
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level('../levels/test_level.data', verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
 
def load_squared_coefs(filename):
  coefs = np.loadtxt(filename).reshape(n_actions, 2*n_features + 1)
  free_coefs = coefs[:,-1]
  coefs = coefs[:,:-1]
  
  for i in xrange(n_actions):
    for j in xrange(2*n_features):
      if j % 2 == 0:
        c[i][j/2] = coefs[i][j]
      else:
        cc[i][(j-1)/2] = coefs[i][j]
    fc[i] = free_coefs[i]
  
 
def calc_reg_for_action(action, state):
  sq_state = np.multiply(state, state)
  
  return np.dot(cc[action], sq_state) + np.dot(c[action], state) + fc[action] + np.dot(ac[action], last_actions)

def update_ac_coefs(i, j, score_diff):
  if score_diff < score_threshold:
    ac[i][j] = ac[i][j] - ac_grads[i][j]
    ac_grads[i][j] = -grad_update * ac_grads[i][j]
  
def run_bbox():
  has_next = 1
  
  load_squared_coefs("coefs_squared.txt")

  with open('acs.bin','rb') as fp:
    loaded_ac = cPickle.load(fp)

  for i in xrange(4):
    for j in xrange(4):
      ac[i][j] = loaded_ac[i][j]
    print ac[i]
  
  
  start = time.time()
  prepare_bbox()
  score = 0
  t = 0
  i = 0
  j = 0
  while has_next:
    if t % step == 0:
      if t != 0:
        score_diff = bbox.get_score() - score
        update_ac_coefs(i, j, score_diff)
        score = bbox.get_score()
        
      if t % (4*step) == 0:
        i = (i + 1) % 4
      j = (j + 1) % 4
      ac[i][j] += ac_grads[i][j]
      
      
    state = bbox.get_state()
    action = get_action_by_state_fast(state)
    has_next = bbox.do_action(action)
    t += 1
  bbox.finish()
  print time.time() - start

