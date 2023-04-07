% Problem Setup
target = 0.99; % The desired prediction rate
n = 8; % # of stages in the PUF

% Initialize the PUF
rng('shuffle');
data = load('weight_diff.txt');
w = zeros(1+n,1);
for i = 1:1:1+n
    randi_offset = randi([1 45480]);
    w(i) = data(randi_offset);
end

% Syntax to query the PUF:
% c = randi([0 1], 1, n); % a random challenge vector
% r = puf_query(c,w);
% you may remove these two lines

% You can use the puf_query function to generate your training dataset
% ADD YOUR DATASET GENERATION CODE HERE
training_size = 100;

X_train = zeros(training_size, n); % initialize the input features of the training dataset
Y_train = zeros(training_size, 1); % initialize the output labels of the training dataset

for i=1:training_size
    c_train = randi([0 1], 1, n); % generate a random challenge vector
    X_train(i,:) = c_train; % add the challenge vector to the input features
    Y_train(i) = puf_query(c_train, w); % calculate the corresponding response and add it to the output labels
end

% Train an SVM
% t0 = cputime;
SVMModel = fitcsvm(X_train,Y_train,'ClassNames',[0,1],'Standardize',true, 'KernelFunction','RBF', 'OptimizeHyperparameters','auto'); % train an SVM model
t1 = cputime;
training_time = t1 - t0;

% Evaluate the Model
n_test = 10000;
correct = 0;
for i=1:1:n_test
    c_test = randi([0 1], 1, n); % generate a random challenge vector
    r = puf_query(c_test,w); % true response
    r0 = predict(SVMModel, c_test); % estimated response using the SVM model
    correct  = correct + (r==r0); % count the correct predictions
end

success_rate = correct/n_test;

% Calculate the Effective Training Time
effective_training_time = training_time;
if success_rate < 0.99
    effective_training_time = training_time + 10000*(0.99-success_rate);
end

% Display Results
disp(['Effective Training Time: ' num2str(effective_training_time) ' seconds']);
disp(['Success Rate: ' num2str(success_rate*100) '%']);
