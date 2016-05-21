import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt, exp
import cPickle

cdef int n_features, n_actions
cdef float coefs[4][37]

for i in xrange(4):
  for j in xrange(37):
    coefs[i][j] = 0

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

def prepare_bbox(name='train', force=False):
    global n_features, n_actions
 
    if bbox.is_level_loaded() and not force:
        bbox.reset_level()
    else:
        bbox.load_level('../levels/'+name+'_level.data', verbose=0)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
 
cdef float calc_reg_for_action(int action, float* state):
  cdef:
    float sum = 0
    int n = 36
    
  for i in xrange(n):
    sum = sum + coefs[action][i] * state[i]
  return sum + coefs[action][36]

def generate_coefs_from_gaussian(means, stds=[[-1]]):
  if stds[0][0] < 0:
    for i in xrange(4):
      for j in xrange(37):
        coefs[i][j] = means[i][j]
  else:
    for i in xrange(4):
      for j in xrange(37):
        coefs[i][j] = float(np.random.normal(means[i][j], stds[i][j], 1)[0])
  
def run_bbox():
  cdef:
    float* state
    int action, has_next = 1
    float score = 0
  
  train_scores = []
  test_scores = []
  
  n_epochs = 100
  n_samples = 100
  n_best = 10
  
  noise = 0.2
  
  means = np.zeros((4, 37))
  stds = np.ones((4, 37))
  
  print 'Loading means and stds'
  with open('ce_means.bin', 'rb') as fp:
      means = cPickle.load(fp)
  with open('ce_stds.bin', 'rb') as fp:
      stds = cPickle.load(fp)

  print 'Training started'
  for epoch in xrange(n_epochs):
    start = time.time()
    
    # Reset samples
    best_samples = np.zeros((n_best, 4, 37))
    best_scores = np.ones(n_best) - 100000
    worst = 0
  
    # Generate samples and save the best
    for t in xrange(n_samples):
    
      # Sample coefs
      generate_coefs_from_gaussian(means, stds)
    
      # Calculate score of this sample
      if t == 0:
        prepare_bbox('train', True)
      else:
        prepare_bbox('train', False)
      while has_next:
        state = bbox.c_get_state()
        action = get_action_by_state_fast(state)
        has_next = bbox.c_do_action(action)
      score = bbox.c_get_score()
      bbox.finish(verbose=0)
      has_next = 1
      
      # Save sample if the score is better than the worst of the saved samples
      if score > best_scores[worst]:
        best_scores[worst] = score
        for i in xrange(4):
          for j in xrange(37):
            best_samples[worst][i][j] = coefs[i][j]
      
      worst = np.argmin(best_scores)
      
    # Update means and stds from the new samples
    means = np.mean(best_samples, axis=0)
    stds = np.std(best_samples, axis=0)
    
    # Add noise
    stds += noise
    
    # Evaluate on test set
    generate_coefs_from_gaussian(means)
    prepare_bbox('test', True)
    while has_next:
      state = bbox.c_get_state()
      action = get_action_by_state_fast(state)
      has_next = bbox.c_do_action(action)
    test_score = bbox.c_get_score()
    bbox.finish(verbose=0)
    has_next = 1
    
    train_scores.append(np.mean(best_scores))
    test_scores.append(test_score)
    
    # Print stuff
    print 'Epoch {:3d} | train score: {:+9.2f} | test score: {:+9.2f} | time: {:.2f}'.format(epoch, train_scores[-1], test_scores[-1], time.time()-start)
    with open('ce_means.bin', 'wb') as fp:
      cPickle.dump(means, fp)
    with open('ce_stds.bin', 'wb') as fp:
      cPickle.dump(stds, fp)
  
    with open('ce_train_scores.bin', 'wb') as fp:
      cPickle.dump(train_scores, fp)
    with open('ce_test_scores.bin', 'wb') as fp:
      cPickle.dump(test_scores, fp)



  