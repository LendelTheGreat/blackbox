import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import cPickle 

srng = RandomStreams()

# Convert into correct type for theano
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

# Weights are shared theano variables
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.1))

# RMSProp to update weights
def RMSprop(cost, params, lr=0.0001, rho=0.999, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

print 'Loading data...'
with open('states.bin', 'rb') as fp:
    data = cPickle.load(fp)

    
print 'Building theano graph...'
# Neural network model, 3 fully connected layers
def model(X, w_h, w_o):
    h = T.round(T.clip(T.dot(X, w_h), 0, 1))
    #h = T.dot(X, w_h)
    y_x = T.dot(h, w_o)
    return h, y_x

# Initialize theano variables for X, Y, and shared variables for weights
X = T.fmatrix()
w_h = init_weights((36, 12))
w_o = init_weights((12, 36))
params = [w_h, w_o]

h, y_x = model(X, w_h, w_o)

# cost = T.mean(T.nnet.categorical_crossentropy(y_x, X)) 
cost = T.sum(T.pow((y_x - X),2))
updates = RMSprop(cost, params)

train = theano.function(inputs=[X], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

print data.shape
# Train in 50 epochs
for i in range(1000):
    # Select minibatch and train
    for start, end in zip(range(0, len(data), 100), range(100, len(data), 100)):
        cost = train(data[start:end])
    # Show test set accuracy. Its cost is not used for optimization,
    # it is just to show progress.
    with open('Encoded_weights.bin','wb') as fp:
        cPickle.dump(w_h.get_value(), fp)

    
    if i % 10 == 0:
        pr_data = predict(data)
        score =  np.mean(np.power((data - pr_data),2))
        print 'Epoch {} score: {:.5f}'.format(i, score)
        
    # In each step save the learned weights
