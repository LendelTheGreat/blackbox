import interface as bbox
cimport interface as bbox
import numpy as np
import time
from random import *
from libc.math cimport sqrt, exp
import cPickle

cdef int n_features, n_actions

cdef float layer_state[50]

cdef float c1[37][50]
cdef float c2[51][4]

cdef int get_action_by_state_fast(float* state):
  cdef:
    int best_act = -1
    float best_val = -1e9
    float val
    
  update_layer_state(state)
    
  for act in xrange(n_actions):
    val = calc_reg_for_action(act)
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
 
cdef void update_layer_state(float* state):
  for i in xrange(50):
    layer_state[i] = 0
    for j in xrange(36):
      layer_state[i] += state[j] * c1[j][i]
    layer_state[i] += c1[j][36]
    
cdef float calc_reg_for_action(int action):
  cdef float sum = 0

  for i in xrange(50):
    sum += layer_state[i] * c2[i][action]
  sum += c2[50][action]
  
  return sum

def generate_coefs_from_gaussian(means1, means2, stds1=[[-1]], stds2=[[-1]]):
  if stds1[0][0] == -1:
    for i in xrange(37):
      for j in xrange(50):
        c1[i][j] = float(means1[i][j])
    for i in xrange(51):
      for j in xrange(4):
        c2[i][j] = float(means2[i][j])
  else:
    for i in xrange(37):
      for j in xrange(50):
        c1[i][j] = float(np.random.normal(means1[i][j], stds1[i][j], 1)[0])
    for i in xrange(51):
      for j in xrange(4):
        c2[i][j] = float(np.random.normal(means2[i][j], stds2[i][j], 1)[0])
  
def run_bbox():
  cdef:
    float* state
    int action, has_next = 1
    float score = 0
  
  train_scores = []
  test_scores = []
  best_test = -100000
  
  n_epochs = 100
  n_samples = 50
  n_best = 5
  
  noise = 0.2
  
  means1 = np.zeros((37, 50))
  means2 = np.zeros((51, 4))
  stds1 = np.ones((37, 50)) * 0.2
  stds2 = np.ones((51, 4))
  
  print 'Loading coefs'
  with open('Encoded_weights_50.bin', 'rb') as fp:
      loaded_c1 = cPickle.load(fp)
  for i in xrange(36):
    for j in xrange(50):
      means1[i][j] = loaded_c1[i][j]
  # with open('ce_means_besttest.bin', 'rb') as fp:
      # means = cPickle.load(fp)
  # with open('ce_stds_besttest.bin', 'rb') as fp:
      # stds = cPickle.load(fp)

  print 'Training started'
  for epoch in xrange(n_epochs):
    start = time.time()
    # Reset samples
    best_samples1 = np.zeros((n_best, 37, 50))
    best_samples2 = np.zeros((n_best, 51, 4))
    best_scores = np.ones(n_best) - 100000
    worst = 0
  
    # Generate samples and save the best
    for t in xrange(n_samples):
    
      # Sample coefs
      generate_coefs_from_gaussian(means1, means2, stds1, stds2)
    
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
        for i in xrange(37):
          for j in xrange(50):
            best_samples1[worst][i][j] = c1[i][j]
        for i in xrange(51):
          for j in xrange(4):
            best_samples2[worst][i][j] = c2[i][j]
      
      worst = np.argmin(best_scores)
      
    # Update means and stds from the new samples
    means1 = np.mean(best_samples1, axis=0)
    means2 = np.mean(best_samples2, axis=0)
    stds1 = np.std(best_samples1, axis=0)
    stds2 = np.std(best_samples2, axis=0)
    
    # Add noise
    stds1 += noise
    stds2 += noise
    
    # Evaluate on test set
    generate_coefs_from_gaussian(means1, means2)
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
    
    if test_scores[-1] > best_test:
      best_test = test_scores[-1]
      with open('ce_means1_besttest.bin', 'wb') as fp:
        cPickle.dump(means1, fp)
      with open('ce_means2_besttest.bin', 'wb') as fp:
        cPickle.dump(means2, fp)
      with open('ce_stds1_besttest.bin', 'wb') as fp:
        cPickle.dump(stds1, fp)
      with open('ce_stds2_besttest.bin', 'wb') as fp:
        cPickle.dump(stds2, fp)
    
    # Print stuff
    print 'Epoch {:3d} | train score: {:+9.2f} | test score: {:+9.2f} | time: {:.2f}'.format(epoch, train_scores[-1], test_scores[-1], time.time()-start)
  
    # with open('ce_train_scores.bin', 'wb') as fp:
      # cPickle.dump(train_scores, fp)
    # with open('ce_test_scores.bin', 'wb') as fp:
      # cPickle.dump(test_scores, fp)



  