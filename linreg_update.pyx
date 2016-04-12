import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt
import cPickle

squared_deep_iterations = 4
lin_deep_iterations = 4
bias_deep_iterations = 4

cdef float cc[4][36]
cdef float c[4][36]
cdef float fc[4]

cdef float cc_grads[4][36]
cdef float c_grads[4][36]
cdef float fc_grads[4]

cdef float grad_update = 0.9
for i in xrange(4):
  for j in xrange(36):
    c_grads[i][j] = 0.002
    cc_grads[i][j] = 0.003
  fc_grads[i] = 0.002

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
        bbox.load_level('../levels/train_level.data', verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
def load_squared_coefs(filename):
  n_actions = 4
  n_features = 36
  coefs = np.loadtxt(filename).reshape(n_actions, 2*n_features + 1)
  free_coefs = coefs[:,-1]
  coefs = coefs[:,:-1]
  
  print coefs.shape
  for i in xrange(n_actions):
    for j in xrange(2*n_features):
      if j % 2 == 0:
        c[i][j/2] = coefs[i][j]
      else:
        cc[i][(j-1)/2] = coefs[i][j]
    fc[i] = free_coefs[i]
 
def load_regression_coefs(filename):
  n_actions = 4
  n_features = 36
  coefs = np.loadtxt(filename).reshape(n_actions, n_features + 1)
  free_coefs = coefs[:,-1]
  coefs = coefs[:,:-1]
  
  for i in xrange(n_actions):
    for j in xrange(n_features):
      c[i][j] = coefs[i][j]
      cc[i][j] = coefs[i][j] * 0.1
    fc[i] = free_coefs[i]
  
 
cdef float calc_reg_for_action(int action, float* state):
  cdef:
    float sum = 0
    int n = 36
    
  for i in xrange(n):
    sum = sum + cc[action][i] * state[i] * state[i] + c[action][i] * state[i]
  return sum + fc[action]

cdef void update_cc_coefs(int i, int j, float score_diff):
  if score_diff < 0:
    cc_grads[i][j] = -grad_update * cc_grads[i][j]
  
cdef void update_c_coefs(int i, int j, float score_diff):
  if score_diff < 0:
    c_grads[i][j] = -grad_update * c_grads[i][j]
    
cdef void update_fc_coefs(int j, float score_diff):
  if score_diff < 0:
    fc_grads[j] = -grad_update * fc_grads[j]

  
def save_coefs():
  with open('coefs_squared_more.txt', 'w') as file:
    for i in xrange(4):
      for j in xrange(36):
        file.write(str(c[i][j]) + '\n')
        file.write(str(cc[i][j]) + '\n')
      file.write(str(fc[i]) + '\n')
 
def run_bbox():
  cdef:
    float* state
    int action, has_next = 1
    float score = 0
    
  load_squared_coefs("coefs_squared.txt")

  scores = []
  
  # Initial score save
  prepare_bbox()
  while has_next:
      state = bbox.c_get_state()
      action = get_action_by_state_fast(state)
      has_next = bbox.c_do_action(action)
  score = bbox.c_get_score()
  scores.append(score)
  bbox.finish()
  has_next = 1
  
  for t in xrange(10):
    print '############ Iteration: {}  Score: {} ############'.format(t, score)
    
    for i in xrange(4):
      for j in xrange(36):
        start = time.time()
        
        for _ in xrange(squared_deep_iterations):
          cc[i][j] += cc_grads[i][j]
          prepare_bbox()

          while has_next:
              state = bbox.c_get_state()
              action = get_action_by_state_fast(state)
              has_next = bbox.c_do_action(action)

          end = time.time()
          score_diff = bbox.c_get_score() - score
          score = bbox.c_get_score()
          
          update_cc_coefs(i, j, score_diff)
          bbox.finish()
          has_next = 1
        
        print 'Time: ' + str(end - start)
        print '##### CC action: {}   state: {}  grad: {} ######'.format(i, j, c_grads[i][j])
        #print 'Score diff: ' + str(score_diff)
    
    for i in xrange(4):
      for j in xrange(36):
        start = time.time()
        
        for _ in xrange(lin_deep_iterations):
          c[i][j] += c_grads[i][j]
          prepare_bbox()

          while has_next:
              state = bbox.c_get_state()
              action = get_action_by_state_fast(state)
              has_next = bbox.c_do_action(action)

          end = time.time()
          score_diff = bbox.c_get_score() - score
          score = bbox.c_get_score()
          
          update_c_coefs(i, j, score_diff)
          bbox.finish()
          has_next = 1
        
        print 'Time: ' + str(end - start)
        print '##### C action: {}   state: {}  grad: {} ######'.format(i, j, c_grads[i][j])
        #print 'Score diff: ' + str(score_diff)
    
    
    for i in xrange(4):
      start = time.time()
      
      for _ in xrange(bias_deep_iterations):
        fc[i] += fc_grads[i]
        prepare_bbox()

        while has_next:
            state = bbox.c_get_state()
            action = get_action_by_state_fast(state)
            has_next = bbox.c_do_action(action)

        end = time.time()
        score_diff = bbox.c_get_score() - score
        score = bbox.c_get_score()
        
        update_fc_coefs(i, score_diff)
        bbox.finish()
        has_next = 1
        
      print 'Time: ' + str(end - start)
      print '##### FC action: {}  grad: {} ######'.format(i, fc_grads[i])
      #print 'Score diff: ' + str(score_diff)
    
    scores.append(score)
  
  print '###### Overall scores'
  for i in xrange(len(scores)):
    print '{} - {}'.format(i, scores[i])
  save_coefs()
  
 



  