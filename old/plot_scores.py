import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 
import cPickle

with open('scores_linreg.bin', 'rb') as fp:
  scores = cPickle.load(fp)
  
fig = plt.figure()
plt.plot(scores)
plt.savefig('scores.jpg')

scores_diff = []  
for i in range(len(scores)-1):
  scores_diff.append(scores[i+1] - scores[i])


fig = plt.figure()
plt.plot(scores_diff)
plt.savefig('scores_diff.png', dpi=300)

