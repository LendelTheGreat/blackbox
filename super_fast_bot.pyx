import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt

cdef float c[4][36]
cdef float fc[4]
cdef float alpha = 0.1
cdef float max = 10.0

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
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
 
def load_regression_coefs(filename):
  coefs = np.loadtxt(filename).reshape(n_actions, n_features + 1)
  free_coefs = coefs[:,-1]
  coefs = coefs[:,:-1]
  
  for i in xrange(n_actions):
    for j in xrange(n_features):
      c[i][j] = coefs[i][j]
    fc[i] = free_coefs[i]
  
 
cdef float calc_reg_for_action(int action, float* state):
  cdef:
    float sum = 0
    int n = 36
    
  for i in xrange(n):
    sum = sum + c[action][i] * state[i]
  return sum + fc[action]
  
cdef int update_reg_coefs(int action, float* state, float score):
  cdef:
    float predicted_score = 0
    
  predicted_score = calc_reg_for_action(action, state)
  
  for i in xrange(36):
    c[action][i] = c[action][i] + alpha * (score - predicted_score) * state[i]
    if c[action][i] > max:
      c[action][i] = max
    elif c[action][i] < -max:
      c[action][i] = -max
  
  fc[action] = fc[action] + alpha * (score - predicted_score)
  if fc[action] > max:
    fc[action] = max
  elif fc[action] < -max:
    fc[action] = -max
      
      
  return 0
  
def save_coefs():
  with open('coefs.txt', 'w') as file:
    for i in xrange(4):
      for j in xrange(36):
        file.write(str(c[i][j]) + '\n')
      file.write(str(fc[i]) + '\n')
 
def run_bbox():
    cdef:
        float* state
        int action, has_next = 1
    
    best = 2882
    
    for i in xrange(1):
      if i % 10 == 0:
        print i
      start = time.time()
      prepare_bbox()
      load_regression_coefs("best_overwrite.txt")
      score = 0
   
      while has_next:
          state = bbox.c_get_state()
          action = get_action_by_state_fast(state)
          has_next = bbox.c_do_action(action)
          
          
          
          # reward = bbox.c_get_score() - score
          # score = bbox.c_get_score()
          # update_reg_coefs(action, state, reward)
          
          
          
   
      end = time.time()
      print 'Time: ' + str(end - start)
      score = bbox.c_get_score()
      if score > best:
        best = score
        print '###################### BEST: ' + str(best)
        
      #print 'Score: ' + str(score)
      bbox.finish()
      has_next = 1
      
    print 'BEST: ' + str(best)
   
        
    