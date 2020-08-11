% Brian Macomber - U25993688

% HOW TO RUN
%   since each part uses data from the part before, it can be run section
%   by section (one after another), or run all at once

clear, clc

load('prostateStnd.mat')

%% 5.4a

[R_Xtrain, C_Xtrain] = size(Xtrain);
[R_Ytrain, C_Ytrain] = size(ytrain);
[R_Xtest, C_Xtest] = size(Xtest);
[R_Ytest, C_Ytest] = size(ytest);

normXtrain = zeros(R_Xtrain, C_Xtrain);
normYtrain = zeros(R_Ytrain, C_Ytrain);
normXtest = zeros(R_Xtest, C_Xtest);
normYtest = zeros(R_Ytest, C_Ytest);

% normalize the Xtrain by finding scaling and offset parameter for each
% column

for i = 1:C_Xtrain
    offset = mean(Xtrain(:,i));
    scale = std(Xtrain(:,i));
    
    normXtrain(:,i) = (Xtrain(:,i) - offset) / scale;
    normXtest(:,i) = (Xtest(:,i) - offset) / scale;
end

offset = mean(ytrain);
scale = std(ytrain);

normYtrain = (ytrain - offset) / scale;
normYtest = (ytest - offset) / scale;



%% 5.4b
% running ridge regression
lambda = exp([-5:10]);

W_ridge = zeros(C_Xtrain, length(lambda));
B_ridge = zeros(1, length(lambda));

for l = 1:length(lambda)
    
    S_x = (1 / C_Xtrain) * (normXtrain') * normXtrain;
    S_xy = (1 / C_Xtrain) * (normXtrain') * normYtrain;
    
    W_ridge(:,l) = (((lambda(l) / R_Xtrain) * eye(C_Xtrain)) + S_x)^-1 * S_xy;
    B_ridge(l) = mean(normYtrain) - (W_ridge(:,l)' * mean(normXtrain)');
    
end


%% 5.4c
[Wrow, Wcol] = size(W_ridge);

figure
for i = 1:Wrow
    plot(log(lambda), W_ridge(i,:))
    
    hold on
    
    
end
xlabel('log (lambda)')
ylabel('Wridge values')
title('Ridge Regression Coefficient vs. values of lambda')
legend('w1','w2','w3','w4','w5','w6','w7','w8')


%% 5.4d

lambda = exp([-5:10]);
[nTrain, ~] = size(normXtrain); % 67 x 8
[nTest, ~] = size(normXtest);

% find MSE of the training data

% find MSE of the test data

% plot the MSE of botg against values of lambda

normMSEtrain = zeros(1, length(lambda));
normMSEtest = zeros(1, length(lambda));

% find MSE of normalized training data
 for l = 1:length(lambda)
    sumVal = 0;
     
    for i = 1:nTrain
        sumVal = sumVal + (normYtrain(i) - ((W_ridge(:,l)') * (normXtrain(i,:)')))^2; 
    end
    
    normMSEtrain(l) = sumVal / nTrain;
 end
 
 % find MSE of noramlized test data
 for l = 1:length(lambda)
    sumVal = 0;
     
    for i = 1:nTest
        sumVal = sumVal + (normYtest(i) - ((W_ridge(:,l)') * (normXtest(i,:)')))^2; 
    end
    
    normMSEtest(l) = sumVal  / nTest;
 end

%plot these MSE vs ln(lambda)
figure
plot(log(lambda), normMSEtrain)
hold on
plot(log(lambda), normMSEtest)
xlabel('ln (lambda) ')
ylabel('MSE')
legend('MSE-Training','MSE-Testing')
title('Mean-Squared-Error of Training and Test sets versus Lambda')

 
 
