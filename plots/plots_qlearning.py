import numpy as np
import matplotlib.pyplot as plt
import cPickle

with open('VISITS.bin', 'rb') as fp:
  states_visited = cPickle.load(fp)

with open('SCORES.bin', 'rb') as fp:
  scores_ql = cPickle.load(fp)
with open('SCORES_random.bin', 'rb') as fp:
  scores_rnd = cPickle.load(fp)
with open('SCORES_action0.bin', 'rb') as fp:
  scores_0 = cPickle.load(fp)
with open('SCORES_action1.bin', 'rb') as fp:
  scores_1 = cPickle.load(fp)
with open('SCORES_action2.bin', 'rb') as fp:
  scores_2 = cPickle.load(fp)
with open('SCORES_action3.bin', 'rb') as fp:
  scores_3 = cPickle.load(fp)
  
  
plt.figure()
plt.plot(scores_ql, label='Q-learning')
plt.plot(scores_rnd, label='Random')
plt.plot(scores_0, label='Action 0', lw=2)
plt.plot(scores_1, label='Action 1', lw=2)
plt.plot(scores_2, label='Action 2', lw=2)
plt.plot(scores_3, label='Action 3', lw=2)
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Q-learning trained on 20 binary features, compared to simple models')
plt.legend()
plt.show()

# states_toplot = states_visited[:10000]
# plt.figure()  
# xs = np.arange(len(states_toplot))
# plt.bar(xs, states_toplot)
# plt.xlabel('State')
# plt.ylabel('Frequency')
# plt.title('Histogram of states visited during Q-learning (first '+str(len(states_toplot))+' states)')
# plt.show()