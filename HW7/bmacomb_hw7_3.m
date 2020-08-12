% HW7 - 7.3
%% load in dataset
clear;clc;

load iris.mat

% seed the random number generation
rng('default');

%parameters go here
t_max = 6000;
lambda = .1;

[N, num_features] = size(X_data_train);

%number of classes
m = 3;

%% 7.3a
% using all of the data
% i - histogram 

X_data_all = [X_data_test; X_data_train];
Y_labels = [Y_label_train; Y_label_test];

figure(1)
histogram(Y_labels, m);
xlabel('Classes');
ylabel('Number of points');
title('Histogram of labels for data set');
ticks = [1 1.85 2.6];
xticks(ticks);
xticklabels({'1','2','3'})

% ii - matrix of empirical correlation coefficients

empCorCoefs = zeros(num_features, num_features);

covmat = cov(X_data_all);

for i = 1:num_features
    for j = 1:num_features
        empCorCoefs(i,j) = covmat(i,j)...
            / sqrt(var(X_data_all(:,i)) * var(X_data_all(:,j)));  
    end
    
end
fprintf('empirical correlation coefficients:\n')
disp(empCorCoefs);

% iii - scatter plots of all pairs of features - 6 unique pairs

k = 1; % subplot iterator
figure(2)
sgtitle('All distinct pairs of features');

for i = 1:num_features
   for j = i:num_features
      if i ~= j
          subplot(3,2,k)
          scatter(X_data_all(:,i), X_data_all(:,j),10,'r*')
          
          title(sprintf('Scatter for feature %d, vs. feature %d',i,j));
          xlabel(sprintf('feature %d',i));
          ylabel(sprintf('feature %d',j));
          
          k = k + 1;
      end   
   end
end

% ********* discuss observations **************





%% ******************** IMPLEMENTATION OF SGD *********************** %%

% initialize big theta to 0
THETA = zeros(num_features + 1 , m);

%7.3b array of g_theta to plot vs t
g_theta_arr = zeros(1, t_max/20);

%7.3c array of training ccr
ccr_train_arr = zeros(1, t_max/20);

%7.3d array of test ccr
ccr_test_arr = zeros(1, t_max/20);

%7.3e array of log loss for test
logLoss_test = zeros(1, t_max/20);


for t = 1:t_max
    % choose j randomly
    j = randi([1 N]);
    
    % compute gradients
    v_k = zeros(num_features + 1 , m);
    
    
    for k = 1:m
        P_kx = 0;
        numerator = 0;
        denom = 0;
        
        % create xj_ext
        xj_ext = [X_data_train(j,:) 1]';
        
        %calc P_kx
        numerator = exp( (THETA(:,k)') * xj_ext);
        
        for l = 1:m
            denom = denom + exp( (THETA(:,l)') * xj_ext);
        end
        
        P_kx = numerator / denom;
        
        % to prevent from taking logs of small #s
        if P_kx < (10^-10)
            P_kx = (10^-10);    
        end
        
        % calc v_k
        v_k(:,k) = (2 * lambda * (THETA(:,k))) + ...
            (N * (P_kx - (k == Y_label_train(j))) * xj_ext); 
    end
    
    % update parameters
    for k = 1:m
       THETA(:,k) = THETA(:,k) - ((.01 / t) * v_k(:,k));
    end
    
    
    % every 20 iterations stuff happens
    if mod(t,20) == 0
        
        % ***** 7.3b calculating g(theta) ****** 
        fz_theta = 0;
        fj_theta = 0;

        % calc f_0(theta)
        for l = 1:m
           fz_theta = fz_theta + (norm(THETA(:,l)) ^ 2); 
        end

        fz_theta = lambda * fz_theta;
        
        fj_theta = zeros(1,N);

        % calc fj_theta
        for i = 1:N
            firstHalf = 0;
            secondHalf = 0;
            
            %calculate the xj_ext for this j
            xj_ext = [X_data_train(i,:) 1]';
            
            for l = 1:m
               firstHalf = firstHalf +  exp( (THETA(:,l)' ) * xj_ext);
            end

            firstHalf = log(firstHalf);

            for l = 1:m
               secondHalf = secondHalf + ( (l == Y_label_train(i)) * ((THETA(:,l)') * xj_ext) );
            end

            fj_theta(i) = (firstHalf - secondHalf);

        end


        % update g_theta vector 
        g_theta_arr(t/20) = fz_theta + sum(fj_theta);
        
        
       % ***** 7.3c calculating ccr of training set *****
       ccr_train = 0;
       ypred_train = zeros(N,1);
       
       for i = 1:N
            [~, argmax] = max(THETA' * [X_data_train(i,:) 1]');  
            if t == t_max
                ypred_train(i) = argmax;
            end
            
            if Y_label_train(i) == argmax
               ccr_train = ccr_train + 1; 
            end
       end
       
       ccr_train_arr(t/20) = ccr_train / N;
       

       % ***** 7.3d calculating ccr of test set *****
       ccr_test = 0;
       [N_test, ~] = size(X_data_test);
       
       ypred_test = zeros(N_test,1);
       
       for i = 1:N_test
            [~, argmax] = max(THETA' * [X_data_test(i,:) 1]');
            if t == t_max
                ypred_test(i) = argmax;
            end
                
            if Y_label_test(i) == argmax
               ccr_test = ccr_test + 1; 
            end
       end
       
       ccr_test_arr(t/20) = ccr_test / N_test;
       
       
       
       
       % ***** 7.3e calculating log loss of the test set *****
       
       P_yj_sum = 0;
       
       for i = 1:N_test
          P_yj_xj_theta = 0;
          top = 0;
          bottom = 0;
           
          % calc P_yj_xj_theta
          xj_ext = [X_data_test(i,:) 1]';
          
          top = exp(THETA(:, Y_label_test(i))' * xj_ext);
          
          for k = 1:m
             bottom = bottom + exp(THETA(:,k)' * xj_ext); 
          end
          
          P_yj_xj_theta = top / bottom;
          
          if P_yj_xj_theta < 10^-10
             P_yj_xj_theta = 10^-10; 
          end
          
          P_yj_sum = P_yj_sum + log(P_yj_xj_theta);
           
       end
       
       
       logLoss_test(t/20) = ((-1) * P_yj_sum) / N_test;
       
       

    end
    
    
end

%% 7.3b
% plot g_theta_arr against t where t = 20t' and t' = 1,2,...,300

t_p = [1:300];
t = 20 * t_p;

%normalize by num of training examples
g_theta_arr_norm = (1 / N) * g_theta_arr;

figure(3)
plot(t, g_theta_arr_norm)
xlabel('Iteration number')
ylabel('Normalized l2-logisitic loss')
title('7.3b - Normlaized Logistic loss vs. iteration number')

%% 7.3c
% plot ccr of training versus t

figure(4)
sgtitle('CCR versus iteration number for Training and Test')

subplot(2,1,1)

plot(t, ccr_train_arr)
xlabel('Iteration number')
ylabel('CCR for training set')
title('7.3c - CCR of training set vs. iteration number')



%% 7.3d
% plot ccr of test versus t

subplot(2,1,2)

plot(t, ccr_test_arr)
xlabel('Iteration number')
ylabel('CCR for test set')
title('7.3d - CCR of test set vs. iteration number')


%% 7.3e
% plot log loss of test versus t

figure(5)

plot(t, logLoss_test)
xlabel('Iteration number')
ylabel('Log loss of test set')
title('7.3e - Log Loss of testing set vs. iteration number')

%% 7.3f
% report final value of THETA
fprintf('\nFinal value of THETA: \n')
disp(THETA)

% report CCR of training
fprintf('\nCCR value for training set: %.3f\n',ccr_train_arr(end))

% report CCR of testing
fprintf('\nCCR value for testing set: %.3f\n',ccr_test_arr(end))

% report training conf mat
confmat_train = confusionmat( ypred_train, Y_label_train);
fprintf('\nConfusion matrix for training set: \n')
disp(confmat_train)

%report test conf mat
confmat_test = confusionmat(ypred_test, Y_label_test);
fprintf('\nConfusion matrix for test set: \n')
disp(confmat_test)


%% 7.3g

%subplot iterator
l = 1;

%theta to work with here
THETA_decision = zeros(3,3);
figure
sgtitle('Visualizing decision boundaries for all the 2D scatter plots')

for i = 1:num_features
   for j = i:num_features
      if i ~= j
          
       % create grid and get dataset from grid
        [Xgrid, Ygrid]=meshgrid([0:.2:10],[0:.2:7]);
         Xdecision = [Xgrid(:), Ygrid(:)];

       [N_decision,  dim_decision] = size(Xdecision);
       
       THETA_decision = [THETA(i,:); THETA(j,:); THETA(num_features + 1, :)];
       
       
          
           % generate deicisons for the meshgrid points in 2d space
           ypred_decision = zeros(N_decision,1);

           for k = 1:N_decision
                [~, argmax] = max(THETA_decision' * [Xdecision(k,:) 1]');  
                ypred_decision(k) = argmax;

           end

          
          subplot(2,3,l);
          hold on
          
          gscatter(Xgrid(:), Ygrid(:), ypred_decision,'rgb');%, ypred_decision);
          xlim([0,10]);
          ylim([0,7]);
          
          scatter(X_data_all(:,i), X_data_all(:,j),10, 'r*');%, ypred_decision);

          title(sprintf('Scatter for feature %d, vs. feature %d',i,j));
          xlabel(sprintf('feature %d',i));
          ylabel(sprintf('feature %d',j));
                
          
          
           l = l + 1;
            hold off
          
      end   
   end
end





