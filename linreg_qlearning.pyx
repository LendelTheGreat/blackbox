import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt, fabs, pow
import cPickle
import math
from libc.stdlib cimport rand, RAND_MAX


cdef int sw = 20
cdef float w[36][20]

for i in xrange(36):
  for j in xrange(sw):
    w[i][j] = rand()/float(RAND_MAX)
    
cdef int state_number_from_raw(float* state):
  cdef:
    float sum
    int bin
    int result = 0
  
  for st in xrange(sw):
    sum = 0
    for i in xrange(36):
      sum += w[i][st]*state[i]
    
    if sum > 0.5:
      bin = 1
    else:
      bin = 0
      
    result += int(pow(2, st) * bin)
    
  return result
  
cdef int choose_action(float eta, int q):
  cdef:
    float best_val = -20000
    int best_act = -1
    float r
    
  r = rand()/float(RAND_MAX)
  
  if r >= eta:
    for a in xrange(4):
      if qs[q][a] > best_val:
        best_val = qs[q][a]
        best_act = a
  else:
    best_act = int((rand()/float(RAND_MAX))*3.9999)
        
  return best_act 
  
# Store state data for plots
states_visited = np.zeros(1048576)

# Initialize qs
cdef float qs[1048576][4]
for st in xrange(1048576):
  for i in xrange(4):
    qs[st][i] = 0.1 * rand()/RAND_MAX
    
# def get_action_greedy(state):
  # #st = binary_from_raw(state)
  # q = state_number_from_raw(state)

  # m = -10000
  # result = np.argmax(qs[q])
  # return (q, result)

# n_features = 0
# n_actions = 0

def prepare_bbox():
  global n_features, n_actions

  if bbox.is_level_loaded():
    bbox.reset_level()
  else:
    bbox.load_level("../levels/train_level.data", verbose=1)
    n_features = bbox.get_num_of_features()
    n_actions = bbox.get_num_of_actions()

def run_bbox():
  cdef:
    int has_next = 1
    float reward
    float immediate_reward
    float mew
    float eta
    float gamma
    float* raw_state
    
  with open('Encoded_weights_20.bin', 'r') as fp:
    weights = np.asarray(cPickle.load(fp))
    
  for i in xrange(36):
    for j in xrange(sw):
      w[i][j] = weights[i][j]

  mew = 0.5
  eta = 0.4
  gamma = 0.95
  
  scores = []

  for iteration in xrange(2000):
    print iteration
    eta *= 0.999
    if eta < 0.05:
      eta = 0.05
    mew  *= 0.999
    if mew < 0.2:
      mew = 0.2
    prepare_bbox()
    reward = 0
    t = 0
    has_next = 1
    raw_state = bbox.c_get_state()
    q = state_number_from_raw(raw_state)
    a = choose_action(eta, q)
    while has_next:
      t += 1
      if has_next:
        has_next = bbox.do_action(a)
      
      immediate_reward = bbox.c_get_score() - reward
      reward = bbox.c_get_score()
      
      raw_state = bbox.c_get_state()
      nq = state_number_from_raw(raw_state)
      na = choose_action(0.0, nq)
      qs[q][a] = (1-mew)*qs[q][a] + mew*(immediate_reward +  gamma * qs[nq][na])
      q = nq
      a = na
      states_visited[q] += 1

    bbox.finish()
    
    
    # Save states visited for histogram plots.
    # with open('VISITS.bin', 'wb') as fp:
      # cPickle.dump(states_visited, fp)
      
    # Save score of each iteration.
    scores.append(bbox.get_score())
    with open('SCORES_moreExploration.bin', 'wb') as fp:
      cPickle.dump(scores, fp)


