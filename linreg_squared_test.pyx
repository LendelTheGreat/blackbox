import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *

cdef float cc[4][36]
cdef float c[4][36]
cdef float fc[4]

cdef int get_action_by_state_fast(float* state):
  cdef:
    int best_act = -1
    float best_val = -1e9
    float val

  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state)
    if val > best_val:
      best_val = val
      best_act = act
 
  return best_act

cdef int n_features, n_actions

def prepare_bbox():
    global n_features, n_actions
 
    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level('../levels/test_level.data', verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
 
def load_squared_coefs(filename):
  n_actions = 4
  n_features = 36
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
  
 
cdef float calc_reg_for_action(int action, float* state):
  cdef:
    float sum = 0
    int n = 36
    
  for i in xrange(n):
    sum = sum + cc[action][i] * state[i] * state[i] + c[action][i] * state[i]
  return sum + fc[action]
 
def run_bbox():
  cdef:
    float* state
    int action, has_next = 1
    
  load_squared_coefs("coefs_squared_more.txt")

  prepare_bbox()
  while has_next:
    state = bbox.c_get_state()
    action = get_action_by_state_fast(state)
    has_next = bbox.c_do_action(action)
  bbox.finish()



  