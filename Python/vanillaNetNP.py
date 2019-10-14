
import numpy as np

# activation: ReLu
def g(x):
    return np.maximum(x, 0.0)

# derivative
def dg(x):
    return np.where(x>0, 1.0, 0.0)

# feed-forward
def forwardPass(x, W1, b1, W2, b2, W3, b3):
    z1 = W1.dot(x) + b1         # op1
    a1 = g(z1)                  # op2
    z2 = W2.dot(a1) + b2        # op3
    a2 = g(z2)                  # op4
    y = W3.dot(a2) + b3         # op5
    
    return [x, z1, a1, z2, a2], y   # we must remember everything

def backwardPass(y, dy, W1, b1, W2, b2, W3, b3, memory):
    x, z1, a1, z2, a2 = memory
    # op5
    dW3 = dy.dot(a2.T)          # param
    db3 = dy                    # param
    da2 = W3.T.dot(dy)          # recurse
    # op4
    dz2 = dg(z2) * da2          # recurse
    # op3
    dW2 = dz2.dot(a1.T)         # param
    db2 = dz2                   # param
    da1 = W2.T.dot(dz2)         # recurse
    # op2
    dz1 = dg(z1) * da1          # recurse
    # op1
    dW1 = dz1.dot(x.T)          # param
    db1 = dz1                   # param
    dx = W1.T.dot(dz1)          # recurse
    
    return dx, dW1, db1, dW2, db2, dW3, db3
    
# test
    
np.random.seed(54321)

x = np.random.normal(size=(5, 1))
W1 = np.random.normal(size=(3, 5))
b1 = np.random.normal(size=(3, 1)) 
W2 = np.random.normal(size=(3, 3))
b2 = np.random.normal(size=(3, 1)) 
W3 = np.random.normal(size=(1, 3))
b3 = np.random.normal(size=(1, 1))

memory, y = forwardPass(x, W1, b1, W2, b2, W3, b3)
print ("y = ", y)

dy = np.array([[1.0]])
dx, dW1, db1, dW2, db2, dW3, db3 = backwardPass(y, dy, W1, b1, W2, b2, W3, b3, memory)
print ("derivs")
print (dx)
print (dW1)
print (db1)
print (dW2)
print (db2)
print (dW3)
print (db3)

    
    