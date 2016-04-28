# What Nantas said
- Build features: autoencoders on 10-100 states, then use them as features for a simple model. 
- Train: perhaps Q-learning idea? 
- 

# To do
- Think about how to implement online learning: learn while you go through the level (i.e. modify weights by responses in the first half, then cash in on the changes in the second half)

- Idea from a random redditor: "I mean it might be able to work. But the fact that it's broken up into "levels" and you're only given one level to train on makes it seem like the train and test "levels" are not iid. Plus they give you a simulation but they dont really let you use it (no monte carlo planning). It also excludes actor-critic since then you're basically just learning the training simulation. So what does that leave? Q-learning and TD both require some sort of model, but the levels are IID. You could probably use a variant that's online but that's probably it."  
https://en.wikipedia.org/wiki/Temporal_difference_learning  
https://en.wikipedia.org/wiki/Q-learning

# Lessons learned so far: 
- Sequences occur: good choice for next action is influenced by the last made choice. However, it is interesting to see that it is not just long sequences of the same action - that would suggest a strong diagonal in this w matrix, which is not present: 

```
[[-0.0335094 , -1.15729392,  1.08289921,  0.93143314],
 [-0.52939415, -0.40155876,  0.30389929, -0.58338237],
 [ 0.18959498, -0.56440568,  0.33902133,  2.06911397],
 [-0.05953684, -0.92938888,  0.26350981,  1.17015839]]
```
