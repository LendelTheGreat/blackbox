import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt, fabs
import cPickle

steps = [80, 100, 150, 200]
thresholds = [2.0, 3.0, 4.0, 5.0]
grad_updates = [0.2, 0.25, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
gradients_ac = [0.15, 0.2, 0.3, 0.4, 0.5]

cdef float score_threshold = 1.0
cdef float grad_update = 0.9
cdef float ac_gradient = 0.11

# RESULTS in score 2570
steps = [100]
thresholds = [3]
grad_updates = [0.25]
gradients_ac = [0.3]

# RESULTS in score 2664.91
# steps = [6000]
# thresholds = [5.0]
# grad_updates = [0.725]
# gradients_ac = [0.2]

cdef float last_actions[4]

cdef float ac[4][4]
cdef float cc[4][36]
cdef float c[4][36]
cdef float fc[4]

cdef float ac_grads[4][4]
cdef float cc_grads[4][36]
cdef float c_grads[4][36]
cdef float fc_grads[4]

for i in xrange(4):
  for j in xrange(36):
    c_grads[i][j] = 0.002
    cc_grads[i][j] = 0.003
  fc_grads[i] = 0.002
  for j in xrange(4):
    ac_grads[i][j] = ac_gradient
  last_actions[i] = 0
  
cdef int get_action_by_state_fast(float* state):
  cdef:
    int best_act = -1
    float best_val = -1e9
    float val
    float sum = 0

  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state)
    if val > best_val:
      best_val = val
      best_act = act
      
  for act in xrange(n_actions):
    last_actions[act] = 0
  last_actions[best_act] = 1
 
  return best_act

cdef int n_features, n_actions

def prepare_bbox(level_name, verbose=0, force=False):
  global n_features, n_actions
  
  if bbox.is_level_loaded() and not force:
    bbox.reset_level()
  else:
    bbox.load_level('../levels/'+level_name+'_level.data', verbose=verbose)
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
  for a in xrange(4):
    sum = sum + last_actions[a] * ac[action][a]
  return sum + fc[action]

cdef void update_ac_coefs(int i, int j, float score_diff):
  if score_diff < score_threshold:
    ac[i][j] = ac[i][j] - ac_grads[i][j]
    ac_grads[i][j] = -grad_update * ac_grads[i][j]
  
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
    
  print '\n######### Loading coefs #########'
  load_squared_coefs("coefs_squared.txt")
  with open('acs.bin','rb') as fp:
    loaded_ac = cPickle.load(fp)
  for i in xrange(4):
    for j in xrange(4):
      ac[i][j] = loaded_ac[i][j]
    print ac[i]

  print '\n######### Obtaining initial scores (train) #########'
  prepare_bbox('train', force=True)
  while has_next:
    state = bbox.c_get_state()
    action = get_action_by_state_fast(state)
    has_next = bbox.c_do_action(action)
  score = bbox.c_get_score()
  bbox.finish()
  has_next = 1
  init_train_score = score
  
  print '\n######################## Start optimizing online ########################'
  best_train_scores = []
  best_test_scores = []
  best_steps = []
  best_thresholds = []
  best_grad_updates = []
  best_ac_grads = []
  
  # Train set
  for s in steps:
    for th in thresholds:
      for gu in grad_updates:
        for acg in gradients_ac:
          step = s
          global score_threshold
          score_threshold = th
          global grad_update
          grad_update = gu
          global ac_gradient
          ac_gradient = acg
    
          train_score = 0
          # Evaluate on train
          for i in xrange(4):
            for j in xrange(4):
              ac[i][j] = loaded_ac[i][j]
              ac_grads[i][j] = ac_gradient
          t = 0
          score = 0
          i = 0
          j = 0
          prepare_bbox('train')
          while has_next:
            if t % step == 0:
              if t != 0:
                score_diff = bbox.c_get_score() - score
                update_ac_coefs(i, j, score_diff)
                score = bbox.c_get_score()
                
              if t % (4*step) == 0:
                i = (i + 1) % 4
              j = (j + 1) % 4
              ac[i][j] += ac_grads[i][j]
              
            state = bbox.c_get_state()
            action = get_action_by_state_fast(state)
            has_next = bbox.c_do_action(action)
            t += 1
          train_score = score - init_train_score
          print 'step={}, thres={}, grad={:.2f}, ac={:.2f}, score={:.2f}'.format(step, score_threshold, grad_update, ac_gradient, train_score)
          bbox.finish()
          has_next = 1
          
          best_train_scores.append(train_score)
          best_steps.append(step)
          best_thresholds.append(score_threshold)
          best_grad_updates.append(grad_update)
          best_ac_grads.append(ac_gradient)
  
  print '\n######### Obtaining initial scores (test) #########'
  prepare_bbox('test', force=True)
  for i in xrange(4):
    for j in xrange(4):
      ac[i][j] = loaded_ac[i][j]
  while has_next:
    state = bbox.c_get_state()
    action = get_action_by_state_fast(state)
    has_next = bbox.c_do_action(action)
  score = bbox.c_get_score()
  bbox.finish()
  has_next = 1
  init_test_score = score
  
  # Test set
  for s in steps:
    for th in thresholds:
      for gu in grad_updates:
        for acg in gradients_ac:
          step = s
          global score_threshold
          score_threshold = th
          global grad_update
          grad_update = gu
          global ac_gradient
          ac_gradient = acg

          test_score = 0
          # Evaluate on test
          for i in xrange(4):
            for j in xrange(4):
              ac[i][j] = loaded_ac[i][j]
              ac_grads[i][j] = ac_gradient
          t = 0
          score = 0
          i = 0
          j = 0
          prepare_bbox('test')
          while has_next:
            if t % step == 0:
              if t != 0:
                score_diff = bbox.c_get_score() - score
                update_ac_coefs(i, j, score_diff)
                score = bbox.c_get_score()
                
              if t % (4*step) == 0:
                i = (i + 1) % 4
              j = (j + 1) % 4
              ac[i][j] += ac_grads[i][j]
              
            state = bbox.c_get_state()
            action = get_action_by_state_fast(state)
            has_next = bbox.c_do_action(action)
            t += 1
          test_score = score - init_test_score
          print 'step={}, thres={}, grad={:.2f}, ac={:.2f}, score={:.2f}'.format(step, score_threshold, grad_update, ac_gradient, test_score)
          bbox.finish()
          has_next = 1
        
          best_test_scores.append(test_score)
          
          
  best_score = -2000000
  for i in xrange(len(best_train_scores)):
    if best_score < best_train_scores[i] + best_test_scores[i]:
      best_score = best_train_scores[i] + best_test_scores[i]
      best_step = best_steps[i]
      best_threshold = best_thresholds[i]
      best_grad_update = best_grad_updates[i]
      best_ac_grad = best_ac_grads[i]

  print '\n######################## Results #########################'
  print 'Best score: {}'.format(best_score)
  print 'Best step: {}'.format(best_step)
  print 'Best threshold: {}'.format(best_threshold)
  print 'Best grad_update: {}'.format(best_grad_update)
  print 'Best ac_grad: {}'.format(best_ac_grad)

  