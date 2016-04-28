import cPickle
import numpy as np
import scipy 

with open('Encoded_weights.bin', 'rb') as fp:
  w = np.asarray(cPickle.load(fp))
  
print w.shape


  
# small_states = states[:100000:1000, :]
# small_mean = np.mean(small_states, axis=0)


# for i, state in enumerate(small_states):
  # small_states[i] = small_states[i] - small_mean
  
  
# print 'small states shape', small_states.shape
# print 'mean', np.mean(small_states)



# U, s, V = np.linalg.svd(small_states, full_matrices=True)
# s = np.diag(s)
# print 'V', V.shape
# print 'U', U.shape

# print 'var expl', np.sum(s[:10])/np.sum(s)
# print 'var expl', np.sum(s[:15])/np.sum(s)
# print 'var expl', np.sum(s[:20])/np.sum(s)

# print s
# with open('svd.bin', 'wb') as fp:
  # cPickle.dump([U,s,V],fp)

