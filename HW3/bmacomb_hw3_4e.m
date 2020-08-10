% EC 414 Introduction to Machine Learning
% Spring semester, 2020
% Homework 3
% by Brian Macomber - U25993688
%
% Problem 4.3 Nearest Neighbor Classifier
% e)





% ----- NOTE ------- %

% This takes about 45 minutes to run, and does not give a correct output
% this was my best attempt, I didn't have time to finish debugging.
% The script runs but the Y_test given is most likely incorrect since it
% returns a CCR of .0679 - and the conf_mat shows this.



clc, clear

fprintf("==== Loading data_mnist_train.mat\n");
load("data_mnist_train.mat");
fprintf("==== Loading data_mnist_test.mat\n");
load("data_mnist_test.mat");

% show test image
% imshow(reshape(X_train(200,:), 28,28)')

% determine size of dataset
[Ntrain, dims] = size(X_train);
[Ntest, ~] = size(X_test);

% precompute components

% Note: To improve performance, we split our calculations into
% batches. A batch is defined as a set of operations to be computed
% at once. We split our data into batches to compute so that the 
% computer is not overloaded with a large matrix.
batch_size = 100;  % fit 4 GB of memory
num_batches = Ntest / batch_size;
curr_Batch = zeros(batch_size, dims);

Y_test = zeros(Ntrain,1);

% Using (x - y) * (x - y)' = x * x' + y * y' - 2 x * y'
for bn = 1:num_batches
  
  % calculate cross term
  
  % compute euclidean distance
  
  batch_start = 1 + (bn - 1) * batch_size;
  batch_stop = batch_start + batch_size - 1;
  
  curr_Batch = X_test(batch_start:batch_stop, :);
  [Nbatch, ~] = size(curr_Batch);
  
  batchDist = zeros(Ntrain, Nbatch);
  
  fprintf("==== Doing 1-NN classification for batch %d\n", bn);
  % find minimum distance for k = 1
  
  for i = 1:Ntrain
      for j = 1:Nbatch
          distSum = 0;
          
%           eucl_Dist = (X_train(i,:) * (X_train(i,:)')) + (X_test(j,:) * (X_test(j,:)')) ...
%               - (2 * (X_train(i,:) * (X_test(j,:)')));
          for k = 1:dims
              distSum = distSum + (X_train(i,k) - curr_Batch(j,k))^2 + ...
                  (2 * X_train(i,k) * curr_Batch(j,k)); 
          end
          batchDist(i,j) = sqrt(distSum);
          
      end
  end
      
  % classify the batch of test points 
  for i = 1:Nbatch
    tempArr = batchDist(:,i);
      %for k = 1 NN no need to loop
    index = find(tempArr == min(tempArr));
    
    Y_test(i + ((bn-1) * batch_size)) = Y_train(index);
    
  end
  
end

% compute confusion matrix
 conf_mat = confusionmat(Y_train(:), Y_test(:));
% compute CCR from confusion matrix

CCR = 0;
for i = 1:10
    CCR = CCR + conf_mat(i,i);
end

CCR = CCR / Ntrain;

fprintf('The CCR is %.4f\n',CCR)


     
