import numpy as np
import cPickle

unique_states = []
for i in xrange(36):
  unique_states.append([])

def load(set):
  print 'Loading '+set+' data...'
  with open('states_'+set+'.bin', 'rb') as fp:
      data = cPickle.load(fp)

  print 'Calculating unique states'
  for st in xrange(len(data)):
    for i in xrange(36):
      if not round(data[st][i], 3) in unique_states[i]:
        unique_states[i].append(round(data[st][i], 3))
        
  # print 'Saving into txt file'
  # with open('states_'+set+'.txt', 'w') as file:
    # for i in xrange(len(data)):
      # str = ''
      # for d in data[i]:
        # str += '{:+.4f} '.format(d)
      # str += '\n'
      # file.write(str)
    
    
load('train')
load('test')

print 'Saving unique states'
with open('unique_states.bin', 'wb') as fp:
  cPickle.dump(unique_states, fp)

print 'Calculating means and stds'
means = []
stds = []
for i in xrange(36):
  means.append(np.mean(unique_states[i]))
  stds.append(np.std(unique_states[i]))

print 'Saving means and stds'
with open('state_means.bin', 'wb') as fp:
  cPickle.dump(means, fp)
with open('state_stds.bin', 'wb') as fp:
  cPickle.dump(stds, fp)
