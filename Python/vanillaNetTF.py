
import numpy as np
import tensorflow as tf

print(tf.__version__)
print(tf.test.is_gpu_available)

# feed-forward
def forwardPassAuto(x, W1, b1, W2, b2, W3, b3):
    # tf syntax not the same as np
    z1 = tf.matmul(W1, x) + b1         
    a1 = tf.nn.relu(z1)                  
    z2 = tf.matmul(W2, a1) + b2        
    a2 = tf.nn.relu(z2)                  
    y = tf.matmul(W3, a2) + b3         
    
    return y   # no need to remember, tf does that behind the scenes

# auto back-prop
tf.reset_default_graph()

# data

np.random.seed(54321)
x = np.random.normal(size=(5, 1))
W1 = np.random.normal(size=(3, 5))
b1 = np.random.normal(size=(3, 1)) 
W2 = np.random.normal(size=(3, 3))
b2 = np.random.normal(size=(3, 1)) 
W3 = np.random.normal(size=(1, 3))
b3 = np.random.normal(size=(1, 1))
target = np.array([[-1.86184893]])

x = tf.constant(x)
W1 = tf.constant(W1)
b1 = tf.constant(b1)
W2 = tf.constant(W2)
b2 = tf.constant(b2)
W3 = tf.constant(W3)
b3 = tf.constant(b3)
target = tf.constant(target)

# set graph
y = forwardPassAuto(x, W1, b1, W2, b2, W3, b3)

# write it
writer = tf.summary.FileWriter('c:/temp/graphs/', tf.get_default_graph())
writer.close()

# automatic back-prop
loss = (y - target) ** 2
grads = tf.gradients(loss, [x, W1, b1, W2, b2, W3, b3])

# compute (on GPU!)
sess = tf.Session()
print ("y = ", sess.run(y))
dx, dW1, db1, dW2, db2, dW3, db3 = sess.run(grads)
print ("derivs")
print (dx)
print (dW1)
print (db1)
print (dW2)
print (db2)
print (dW3)
print (db3)
sess.close()
    
    