% EC 414 Introduction to Machine Learning
% Spring semester, 2020
% Homework 3
%
% Problem 4.3 Nearest Neighbor Classifier
% a), b), c), and d)

clc, clear

fprintf("==== Loading data_knnSimulation.mat\n");
load("data_knnSimulation.mat")

Ntrain = size(Xtrain,1);

%% a) Plotting
% include a scatter plot
% MATLAB function: gscatter()

gscatter(Xtrain(:,1), Xtrain(:,2), ytrain);

% label axis and include title
xlabel('X1')
ylabel('X2')
title('kNN Classification in 2D')


%% b)Plotting Probabilities on a 2D map
K = 10;
% specify grid
[Xgrid, Ygrid]=meshgrid([-3.5:0.1:6],[-3:0.1:6.5]);
Xtest = [Xgrid(:),Ygrid(:)];
[Ntest,dim]=size(Xtest);

% compute probabilities of being in class 2 and class 3 for each point on grid

% preallocations of used memory
probabilities_class2 = zeros(Ntest, 1);
probabilities_class3 = zeros(Ntest, 1);
distances = zeros(Ntrain, Ntest);
tempArr = zeros(Ntrain,1);
Ytest = zeros(Ntest,1);
yTest_k = zeros(K,1);

% --------------Running kNN Classification----------------

% Compute all the distances between training and test points
for i = 1:Ntrain
    for j = 1:Ntest
        distances(i,j) = sqrt((Xtrain(i,1) - Xtest(j,1))^2 + (Xtrain(i,2) - Xtest(j,2))^2);
    end
end

ysum_class2 = 0;
ysum_class3 = 0;

% find the kth smallest elements and their labels
% average the labels to find the corresponding test label value
for i = 1:Ntest
   tempArr = distances(:,i);
   for k = 1:K
       
       index = find(tempArr == min(tempArr));
       yTest_k(k) = ytrain(index);
       if yTest_k(k) == 2
          ysum_class2 = ysum_class2+1; 
       end
       if yTest_k(k) == 3
          ysum_class3 = ysum_class3+1; 
       end
       %set this index to something that won't be seen as min
       tempArr(index) = inf;
       
   end
   
   probabilities_class2(i) = ysum_class2/K;
   probabilities_class3(i) = ysum_class3/K;

   ysum_class2 = 0;
   ysum_class3 = 0;
   
end


% Figure for class 2
figure
class2ProbonGrid = reshape(probabilities_class2,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
%remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('Probabilities of Ylabel being class 2')
% axis([-4 4 -3 6]);

% Figure for class 3
figure
class2ProbonGrid = reshape(probabilities_class3,size(Xgrid));
contourf(Xgrid,Ygrid,class2ProbonGrid);
colorbar;
%remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('Probabilities of Ylabel being class 3')
% axis([-4 4 -3 6]);



%% c) Class label predictions
K = 1 ; % K = 1 case

% --------------Running kNN Classification----------------
distances = zeros(Ntrain, Ntest);
tempArr = zeros(Ntrain,1);
yTest_k = zeros(K,1);
Ytest = zeros(Ntest,1);


% Compute all the distances between training and test points
for i = 1:Ntrain
    for j = 1:Ntest
            distances(i,j) = sqrt((Xtrain(i,1) - Xtest(j,1))^2 + ...
                (Xtrain(i,2) - Xtest(j,2))^2);
    end
end

ysum = 0;

% find the kth smallest elements and their labels
% average the labels to find the corresponding test label value
for i = 1:Ntest
   tempArr = distances(:,i);
   for k = 1:K
       
       index = find(tempArr == min(tempArr));
       yTest_k(k) = ytrain(index);
       %set this index to something that won't be seen as min
       tempArr(index) = inf;
       
   end
   
   %NN classification outputs most frequent label of k nearest neighbors
   Ytest(i) = mode(yTest_k);
   
end

% compute predictions
ypred_k1 = Ytest;
figure
gscatter(Xgrid(:),Ygrid(:),ypred_k1,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('kNN classification with k = 1')

% repeat steps above for the K=5 case. Include code for this below.

K = 5 ; % K = 5 case

% --------------Running kNN Classification----------------
distances = zeros(Ntrain, Ntest);
tempArr = zeros(Ntrain,1);
yTest_k = zeros(K,1);
Ytest = zeros(Ntest,1);


% Compute all the distances between training and test points
for i = 1:Ntrain
    for j = 1:Ntest
         distances(i,j) = sqrt((Xtrain(i,1) - Xtest(j,1))^2 + ...
            (Xtrain(i,2) - Xtest(j,2))^2);
    end
end

ysum = 0;

% find the kth smallest elements and their labels
% average the labels to find the corresponding test label value
for i = 1:Ntest
   tempArr = distances(:,i);
   for k = 1:K
       
       index = find(tempArr == min(tempArr));
       yTest_k(k) = ytrain(index);
       %set this index to something that won't be seen as min
       tempArr(index) = inf;
       
   end
   
   %NN classification outputs most frequent label of k nearest neighbors
   Ytest(i) = mode(yTest_k);
   
end

% compute predictions
ypred_k5 = Ytest;
figure
gscatter(Xgrid(:),Ygrid(:),ypred_k5,'rgb')
xlim([-3.5,6]);
ylim([-3,6.5]);
% remember to include title and labels!
xlabel('X1')
ylabel('X2')
title('kNN classification with k = 5')


%% d) LOOCV CCR computations

validDists = zeros(Ntrain, Ntrain);
tempDist = zeros(Ntrain,1);
ypred = zeros(Ntrain,1);

for k = 1:2:11
    % Find distance from test train point to all other training points
    validY = zeros(1,k);
    for i = 1:Ntrain
       for j = 1:Ntrain
            if i ~= j
                validDists(i,j) = sqrt((Xtrain(i,1) - Xtrain(j,1))^2 + (Xtrain(i,2) - Xtrain(j,2))^2);
            else
                validDists(i,j) = inf;
            end
       end
    end

    
   % using k , classify test point and add to ypred(i)
   for i = 1:Ntrain
      tempDist = validDists(:,i);
      
      for x = 1:k 
          index = find(tempDist == min(tempDist));
          validY(x) = ytrain(index);
          tempDist(index) = inf;
      end
      
      ypred(i) = mode(validY);
   end


    % compute confusion matrix
     conf_mat = confusionmat(ytrain(:), ypred(:));
    % from confusion matrix, compute CCR
    CCR = 0;
     for i = 1:3
         CCR = CCR + conf_mat(i,i);
     end
     CCR = CCR / Ntrain;
%      fprintf('for k=%d, CCR is %.3f\n',k,CCR)
     
%      CCR_values = zeros(1,6);
    % below is logic for collecting CCRs into one vector
    if k == 1
        CCR_values = CCR;
    else
        CCR_values = [CCR_values, CCR];
    end

    validDists = zeros(Ntrain, Ntrain);
    tempDist = zeros(Ntrain,1);
    ypred = zeros(Ntrain,1);
end

% plot CCR values for k = 1,3,5,7,9,11
% label x/y axes and include title
k = 1:2:11;
plot(k, CCR_values, 'b*')
% axis([1 11 .7 .9])
xlabel('k');
ylabel('Correctly Classified Ratio');
title('k vs. CCR: Choosing the best k');



