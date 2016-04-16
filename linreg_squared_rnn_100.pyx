import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt, fabs
import cPickle

recurrent_deep_iterations = 5
cdef float last_actions[4]

cdef int memory_length = 5
cdef float ac_memory[4][1258935]
cdef float ac_memory_coefs[4][4]
cdef float ac_memory_grads[4][4]

cdef float ac[4][4]
cdef float cc[4][36]
cdef float c[4][36]
cdef float fc[4]

cdef float ac_grads[4][4]
cdef float cc_grads[4][36]
cdef float c_grads[4][36]
cdef float fc_grads[4]

cdef float grad_update = 0.9
for i in xrange(4):
  for j in xrange(36):
    c_grads[i][j] = 0.002
    cc_grads[i][j] = 0.003
  fc_grads[i] = 0.002
  for j in xrange(4):
    ac_grads[i][j] = 0.5
    if i == j:
      ac[i][j] = 1
    else:
      ac[i][j] = 0
  last_actions[i] = 0
  
# Initializes memory (array of last 100 actions) as zeros
# and corresponding coeffs & grads
for i in xrange(4):
  for j in xrange(1258935):
    ac_memory[i][j] = 0
  for j in xrange(4):
    ac_memory_grads[i][j] = 0.25
    if i == j:
      ac_memory_coefs[i][j] = 0.5
    else:
      ac_memory_coefs[i][j] = 0.0

cdef int get_action_by_state_fast(float* state, int level):
  cdef:
    int best_act = -1
    float best_val = -1e9
    float val
    float sum = 0
    int max_a
    
  # Get memory term
  max_a = max_ac_memory(level)
  
  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state, max_a)
    if val > best_val:
      best_val = val
      best_act = act
      
  for act in xrange(n_actions):
    last_actions[act] = 0
  last_actions[best_act] = 1
  
  # Update memory with best chosen action
  update_ac_memory(best_act, level)
  
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
  
 
cdef float calc_reg_for_action(int action, float* state, int max_a):
  cdef:
    float sum = 0
    int n = 36
  
  for i in xrange(n):
    sum = sum + cc[action][i] * state[i] * state[i] + c[action][i] * state[i]
  for a in xrange(4):
    sum = sum + last_actions[a] * ac[action][a]
  # Add memory term
  sum = sum + ac_memory_coefs[action][max_a]
  return sum + fc[action]

  
cdef void update_ac_memory_coefs(int i, int j, float score_diff):
  if score_diff < 0:
    ac_memory_grads[i][j] = -grad_update * ac_memory_grads[i][j]
  
cdef void update_ac_coefs(int i, int j, float score_diff):
  if score_diff < 0:
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


cdef void update_ac_memory(int action, int level):
  """
  Appends last action into memory 
  """
  ac_memory[action][level] = 1

cdef int max_ac_memory(int level):
  """
  @Arg: last action
  Calculates which action occured most times in last 100 rounds
  """
  cdef float sum[4]
  cdef int max_a = 0
  for i in xrange(4):
    sum[i] = 0
    for j in xrange(level, level + memory_length):
      sum[i] = sum[i] + ac_memory[i][j]
    if sum[max_a] < sum[i]:
      max_a = i
  return max_a
  
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
    int level
    
  load_squared_coefs("coefs_squared.txt")
  with open('acs.bin','rb') as fp:
    loaded_ac = cPickle.load(fp)

  for i in xrange(4):
    for j in xrange(4):
      ac[i][j] = loaded_ac[i][j]
    
    
  scores = []
  
  # Initial score save
  prepare_bbox()
  while has_next:
    state = bbox.c_get_state()
    level = bbox.c_get_time()
    action = get_action_by_state_fast(state, level)
    has_next = bbox.c_do_action(action)
  score = bbox.c_get_score()
  scores.append(score)
  bbox.finish()
  has_next = 1
  
  if True:
    for t in xrange(4):
      print '############ Iteration: {}  Score: {} ############'.format(t, score)
      
      for i in xrange(4):
        for j in xrange(4):
          start = time.time()
          
          for _ in xrange(recurrent_deep_iterations):
            # ac[i][j] += ac_grads[i][j]
            ac_memory_coefs[i][j] = ac_memory_coefs[i][j] + ac_memory_grads[i][j]
            prepare_bbox()

            while has_next:
              state = bbox.c_get_state()
              level = bbox.c_get_time()
              action = get_action_by_state_fast(state, level)
              has_next = bbox.c_do_action(action)

            end = time.time()
            score_diff = bbox.c_get_score() - score
            score = bbox.c_get_score()
            
            #update_ac_coefs(i, j, score_diff)
            update_ac_memory_coefs(i, j, score_diff)
            bbox.finish()
            has_next = 1
          
          print 'Time: ' + str(end - start)
          print '##### AC action: {}   state: {}  grad: {} ######'.format(i, j, ac_memory_grads[i][j])
      for pr in xrange(4):
        for pr2 in xrange(4):
          print ac_memory_coefs[pr][pr2]
          #print 'Score diff: ' + str(score_diff)
      
      
    scores.append(score)
    
    amc = np.zeros((4,4))
    for pr in xrange(4):
        for pr2 in xrange(4):
          amc[pr][pr2] = ac_memory_coefs[pr][pr2]
    print 'Saving results'
    with open('acs_mem_3.bin','wb') as fp:
      cPickle.dump(amc,fp)
    
  
  print '###### Overall scores'
  for i in xrange(len(scores)):
    print '{} - {}'.format(i, scores[i])
    
  
  
 



  