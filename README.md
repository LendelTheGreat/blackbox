# To do
- Think about how to implement online learning: learn while you go through the level (i.e. modify weights by responses in the first half, then cash in on the changes in the second half)

# Lessons learned so far: 
- Sequences occur: good choice for next action is influenced by the last made choice. However, it is interesting to see that it is not just long sequences of the same action - that would suggest a strong diagonal in this w matrix, which is not present: 
array([[-0.0335094 , -1.15729392,  1.08289921,  0.93143314],
       [-0.52939415, -0.40155876,  0.30389929, -0.58338237],
       [ 0.18959498, -0.56440568,  0.33902133,  2.06911397],
       [-0.05953684, -0.92938888,  0.26350981,  1.17015839]])

