import interface as bbox
import numpy as np
import time
from random import *
import cPickle
ac = np.zeros((4, 4))

cc = np.zeros((4, 36))
c = np.zeros((4, 36))
fc = np.zeros(4)
ac_memory_coefs = np.zeros((4,4))
memory_length = 1000

ac_memory = np.zeros((4,1258935))
last_actions = np.zeros(4)

def get_action_by_state_fast(state, level):
  best_act = -1
  best_val = -1e9
  max_a = max_ac_memory(level)
  for act in xrange(n_actions):
    val = calc_reg_for_action(act, state, max_a)
    if val > best_val:
      best_val = val
      best_act = act
 
  for act in xrange(n_actions):
    last_actions[act] = 0
  last_actions[best_act] = 1
  
  update_ac_memory(level)
  
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
  
def update_ac_memory(level):
  """
  Appends last action into memory 
  """
  global ac_memory
  ac_memory[:,level] = last_actions

def max_ac_memory(level):
  """
  @Arg: level
  Calculates which action occured most times in last 100 rounds
  """
  sum = np.zeros(4)
  max_a = 0
  for i in xrange(4):
    sum[i] = np.sum(ac_memory[i,level-memory_length:level])
    if sum[max_a] < sum[i]:
      max_a = i
  return max_a

  
def calc_reg_for_action(action, state, max_a):
  sq_state = np.multiply(state, state)
  global ac_memory_coefs
  return np.dot(cc[action], sq_state) + np.dot(c[action], state) + fc[action] + np.dot(ac[action], last_actions) + ac_memory_coefs[action][max_a]
 
def run_bbox():
  has_next = 1
  
  load_squared_coefs("coefs_squared.txt")

  with open('acs.bin','rb') as fp:
    loaded_ac = cPickle.load(fp)

  for i in xrange(4):
    for j in xrange(4):
      ac[i][j] = loaded_ac[i][j]
    
  with open('acs_mem_3.bin','rb') as fp:
    loaded_ac_m = cPickle.load(fp)
  
  global ac_memory_coefs
  for i in xrange(4):
    for j in xrange(4):
      ac_memory_coefs[i][j] = loaded_ac_m[i][j]
    
  
  start = time.time()
  prepare_bbox()
  while has_next:
    state = bbox.get_state()
    level = bbox.get_time()
    action = get_action_by_state_fast(state, level)
    has_next = bbox.do_action(action)
  bbox.finish()
  print time.time() - start



  