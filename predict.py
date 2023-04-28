import numpy as np
import time
#logistic regression
from sklearn.linear_model import LogisticRegression

def transform_X(X):
    n = X.shape[1]
    phi_X = np.ones((X.shape[0], n+1))
    phi_X[:, n] = 1
    for i in range(n-1, -1, -1):
        phi_X[:, i] = (2*X[:, i]-1)*phi_X[:, i+1]
    return phi_X


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

# Syntax to query the PUF:
c = np.random.randint(0, 2, size=(1, n))  # a random challenge vector
r = puf_query(c, w)
# you may remove these two lines

# You can use the puf_query function to generate your training dataset
# ADD YOUR TRAINING CODE HERE

# Generate the training dataset
print("Generating training set...")
training_size = 4500
X = np.random.randint(0, 2, size=(training_size, n)) 
y = np.zeros((training_size, 1))
for i in range(training_size):
    y[i] = puf_query(X[i:i+1, :], w)

print("Training set generated")

# Train the decision tree
t0 = time.process_time()
dt = LogisticRegression(solver='lbfgs', max_iter=1000)
X = transform_X(X)
dt.fit(X, y.ravel())
t1 = time.process_time()
training_time = t1 - t0
print("Training time:", training_time)

# Evaluate the decision tree
n_test = 10000
correct = 0
for i in range(1, n_test+1):
    c_test = np.random.randint(0, 2, size=(1, n))  # a random challenge vector
    r = puf_query(c_test, w)
    c_test = transform_X(c_test)
    r_dt = dt.predict(c_test)
    correct += (r==r_dt)

success_rate = correct/n_test
print("Success rate:", success_rate)

# If the success rate is less than 99%, a penalty time will be added
# One second is add for each 0.01% below 99%.
effective_training_time = training_time
if success_rate < 0.99:
    effective_training_time = training_time + 10000*(0.99-success_rate)
print("Effective training time:", effective_training_time)
