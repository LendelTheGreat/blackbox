import interface as bbox
import numpy as np
import time
from random import *

cc = np.zeros((4, 36))
c = np.zeros((4, 36))
fc = np.zeros(4)

def get_action_by_state_fast(state):
  best_act = -1
  best_val = -1e9

  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state)
    if val > best_val:
      best_val = val
      best_act = act
 
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
  return np.dot(cc[action], sq_state) + + np.dot(c[action], state) + fc[action]
 
def run_bbox():
  has_next = 1
  
  load_squared_coefs("coefs_squared.txt")

  start = time.time()
  prepare_bbox()
  while has_next:
    state = bbox.get_state()
    action = get_action_by_state_fast(state)
    has_next = bbox.do_action(action)
  bbox.finish()
  print time.time() - start



  