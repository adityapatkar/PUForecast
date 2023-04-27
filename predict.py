import numpy as np
import time

def puf_query(c, w):
    n = c.shape[1]
    phi = np.ones(n+1)
    phi[n] = 1
    for i in range(n-1, -1, -1):
        phi[i] = (2*c[0,i]-1)*phi[i+1]
    r = (np.dot(phi, w) > 0)
    return r
    
# Problem Setup
target = 0.99  # The desired prediction rate
n = 64  # number of stages in the PUF

# Initialize the PUF
np.random.seed(int(time.time()))
data = np.loadtxt('weight_diff.txt')
w = np.zeros((n+1, 1))
for i in range(1, n+2):
    randi_offset = np.random.randint(1, 45481)
    w[i-1] = data[randi_offset-1]

# You can use the puf_query function to generate your training dataset
# ADD YOUR DATASET GENERATION CODE HERE

print("Generating training set...")
training_size = 10000
X = np.random.randint(0, 2, size=(training_size, n)) 
y = np.zeros((training_size, 1))
for i in range(training_size):
    y[i] = puf_query(X[i:i+1, :], w)

print("Training set generated")

# ADD YOUR TRAINING CODE HERE
w0 = np.zeros((n+1, 1)) # The estimated value of w.
t0 = time.process_time()
learning_rate = 0.1
for i in range(training_size):
    c = X[i:i+1,:]
    phi = np.ones(n+1)
    phi[n] = 1
    for j in range(n-1, -1, -1):
        phi[j] = (2*c[0,j]-1)*phi[j+1]
    z = np.dot(phi, w0)
    y_pred = (z > 0)
    error = y[i][0] - y_pred
    delta = learning_rate * error * phi.reshape((n+1,1))
    w0 = w0 + delta

t1 = time.process_time()
training_time = t1 - t0  # time taken to get w0
print("Training time:", training_time)
print("Training size:", training_size)

# Evaluate your result
n_test = 10000
correct = 0
for i in range(1, n_test+1):
    c_test = np.random.randint(0, 2, size=(1, n))  # a random challenge vector
    #convert 0 to -1
    r = puf_query(c_test, w)
    r0 = puf_query(c_test, w0)
    correct += (r==r0)

success_rate = correct/n_test
print("Success rate:", success_rate)

# If the success rate is less than 99%, a penalty time will be added
# One second is add for each 0.01% below 99%.
effective_training_time = training_time
if success_rate < 0.99:
    effective_training_time = training_time + 10000*(0.99-success_rate)
print("Effective training time:", effective_training_time)



