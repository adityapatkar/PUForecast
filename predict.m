 % Problem Setup
target = 0.99; % The desired prediction rate
n = 64; % # of stages in the PUF

% Initialize the PUF 
rng('shuffle');
data = load('weight_diff.txt');
w = zeros(1+n,1);
for i = 1:1:1+n
    randi_offset = randi([1 45480]);
    w(i) = data(randi_offset);
end

%testing loop

%params
training_size = 8000;
repeat = 50;
lrate = .0001;

    
training_set = [];
training_res = [];
phi_set = [];
for i = 1:training_size
    c = randi([0 1], 1, n);
    r = puf_query(c,w);
    [~,n] = size(c);
    phi = ones(1,n+1);
    phi(n+1) = 1;
    for i = n:-1:1
	    phi(i) = (2*c(i)-1)*phi(i+1);
    end
    training_set = [training_set; c];
    phi_set = [phi_set; phi];
    training_res = [training_res; r];
end


w0 = zeros(1+n,1); % The estimated value of w.
t0 = cputime;

%gradient descent algorithm call
[w0, costs] = gradDesc(repeat,lrate,w0,phi_set,training_res,length(training_res));

t1 = cputime;
training_time = t1 - t0; % time taken to get w0

% Evaluate your result
n_test = 10000;
correct = 0;
for i=1:1:n_test
    c_test = randi([0 1], 1, n); % a random challenge vector
    r = puf_query(c_test,w);
    r0 = puf_query(c_test,w0);
    correct  = correct + (r==r0);
end

success_rate = correct/n_test;

% If the success rate is less than 99%, a penalty time will be added
% One second is add for each 0.01% below 99%.
effective_training_time = training_time;
if success_rate < 0.99
    effective_training_time = training_time + 10000*(0.99-success_rate);
end

disp('Success_rate');
disp(success_rate);
disp('Effective training time');
disp(effective_training_time);