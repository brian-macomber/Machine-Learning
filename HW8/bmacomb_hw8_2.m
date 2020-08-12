% 8.2

%% load in dataset
clear;clc;

load iris.mat

% seed the random number generation
rng('default');

%% separate data set

% only x2 and x4
X_train = [X_data_train(:,2) X_data_train(:,4)];
X_test = [X_data_test(:,2) X_data_test(:,4)];



%% define parameters
t_max = 2e5;
C = 1.2;

[~, d] = size(X_train);
m = 3; % number of classes

%%  classes 1 and 2 - separate data

BIG_THETA = zeros(d + 1, 3);

z = 3;
a = 1;

for k = 1:m
    for l = k:m
        if k ~= l 
            % SPLIT training set
            X_curr_train = X_train(Y_label_train ~= z, :);
            X_curr_test = X_test(Y_label_test ~= z, :);

            Y_curr_train = Y_label_train(Y_label_train ~= z);
            Y_curr_test = Y_label_test(Y_label_test ~= z);

            [N_curr, ~] = size(X_curr_train);


            % IMPLEMENTATION OF STOCHASTIC SGD FOR SOFT-MARGIN BINARY SVM

            THETA = zeros(d + 1, 1);

            ccr_train_arr = zeros(1, t_max / 1000);

            ccr_test_arr = zeros(1, t_max / 1000);

            g_arr = zeros(1, t_max / 1000);
            
            v = zeros(d + 1, 1);

            for t = 1:t_max
                % choose j randomly
                j = randi([1 N_curr]);

                % compute xj_ext
                xj_ext = [X_curr_train(j,:)' ; 1];

                v = [THETA(1:d, 1); 0];
                
                if Y_curr_train(j) == k
                    currIndex = 1;
                else
                    currIndex = -1;
                end
                

                if ((currIndex * THETA' * xj_ext) < 1)
                    v = v - (N_curr * C * currIndex * xj_ext);
                end

                % update parameters
                THETA = THETA - ((.5 / t) * v);

                % for every 1000 iterations
                if (mod(t,1000) == 0 )

                    % part a - find g(theta)

                    % compute f_0
                    f_0 = .5 * (norm(THETA(1:d, 1))^2);

                    %copmute f_j
                    f_j = 0;

                    for i = 1:N_curr
                        % compute x_ext
                        x_ext = [X_curr_train(i,:)' ; 1];
                        
                        if Y_curr_train(i) == k
                            currIndex = 1;
                        else
                            currIndex = -1;
                        end

                        %add to f_j
                        f_j = f_j + (C * max([0 (1 - (currIndex * THETA' * x_ext))]));
                    end

                    g_arr(t/1000) = f_0 + f_j;


                   % part b - training CCR

                   ccr_train = 0;
                   ypred_train = zeros(N_curr,1);

                   for i = 1:N_curr
                       % decision rule
                       x_ext = [X_curr_train(i,:)' ; 1];

                       ypred = sign(THETA' * x_ext);
                       
                       if ypred > 0
                           ypred = k;
                       else
                           ypred = l; 
                       end

                        if Y_curr_train(i) == ypred
                           ccr_train = ccr_train + 1; 
                        end
                        
                        if t == t_max
                            ypred_train(i) = ypred;
                        end
                   end

                   ccr_train_arr(t/1000) = ccr_train / N_curr;

                   % part c - test CCR

                   ccr_test = 0;
                   [N_test, ~] = size(X_curr_test);
                   ypred_test = zeros(N_test,1);

                   for i = 1:N_test
                       % decision rule
                       x_ext = [X_curr_test(i,:)' ; 1];

                       ypred = sign(THETA' * x_ext);
                       
                       if ypred > 0
                           ypred = k;
                       else
                           ypred = l; 
                       end 

                        if Y_curr_test(i) == ypred
                           ccr_test = ccr_test + 1; 
                        end

                        if t == t_max
                            ypred_test(i) = ypred;
                        end
                   end

                   ccr_test_arr(t/1000) = ccr_test / N_test;


                end


            end % end t_max
            

            % plotting 
            figure
            sgtitle(sprintf('8.2 a b and c for classes %d and %d',k,l))
            t_p = [1:200];
            t = t_p * 1000;

            % part a
            subplot(1,3,1)

            g_arr_norm = g_arr / N_curr;

            plot(t,g_arr_norm,'r')
            xlabel('iteration number')
            ylabel('sample-normalized cost')
            title('8.2a: Sample normalized cost vs. iteration number')
            axis([0 2e5 0 1])




            % part b
            subplot(1,3,2)


            plot(t, ccr_train_arr,'r')
            xlabel('iteration number')
            ylabel('CCR for training data')
            title('8.2b: CCR for training data vs. iteration number')
            axis([0 2e5 0 1])

            % part c
            subplot(1,3,3)

            plot(t, ccr_test_arr,'r')
            xlabel('iteration number')
            ylabel('CCR for testing data')
            title('8.2c: CCR for testing data vs. iteration number')
            axis([0 2e5 0 1])
            
            
            % printing for part d
            fprintf('\n\nValues for classes %d and %d: \n',k,l) 
            fprintf('The final value for Theta is: \n')
            disp(THETA)
            
            fprintf('training CCR is: %.2f\n',ccr_train_arr(end))
            fprintf('testing CCR is: %.2f\n',ccr_test_arr(end))
            
            % report training conf mat
            confmat_train = confusionmat(ypred_train, Y_curr_train);
            fprintf('\nConfusion matrix for training set: \n')
            disp(confmat_train)

            %report test conf mat
            confmat_test = confusionmat(ypred_test, Y_curr_test);
            fprintf('\nConfusion matrix for test set: \n')
            disp(confmat_test)
            
            BIG_THETA(:,a) = THETA;
            
            z = z - 1;
            a = a + 1;
            
            
            
            
        end
    end
end


% part e
% Training 
  
    

[N, ~] = size(X_train);
final_y_pred_train = zeros(1,N);

finalClassifier_train = zeros(105, 3);

ccrtrain = 0;

for j = 1:N
    
    X_extern = [X_train(j,:) 1]';

    
    
    for i = 1:m
        if i == 1
            b = 1;
            c = 2;
        elseif i == 2
            b = 1;
            c = 3;
        else
            b = 2;
            c = 3;
        end
        
       pred = sign(BIG_THETA(:,i)' * X_extern);
       
       if pred == 1
           finalClassifier_train(j,b) = finalClassifier_train(j,b) + 1;
       else
           finalClassifier_train(j,c) = finalClassifier_train(j,c) + 1;
       end
       
        
    end
    
    [~,final_y_pred_train(j)] = max(finalClassifier_train(j,:));
    
    %ccr train
    if final_y_pred_train(j) == Y_label_train(j)
        ccrtrain = ccrtrain + 1;
    end
    
    
end
ccrtrain = ccrtrain / N;

% Test
[N_test, ~] = size(X_test);
final_y_pred_test = zeros(1,N_test);

finalClassifier_test = zeros(N_test,m);

 ccrtest = 0;

for j = 1:N_test
   
    X_extern = [X_test(j,:) 1]';
    
    for i = 1:m
        if i == 1
            b = 1;
            c = 2;
        elseif i == 2
            b = 1;
            c = 3;
        else
            b = 2;
            c = 3;
        end
        
       pred = sign(BIG_THETA(:,i)' * X_extern);
       
       if pred == 1
           finalClassifier_test(j,b) = finalClassifier_test(j,b) + 1;
       else
           finalClassifier_test(j,c) = finalClassifier_test(j,c) + 1;
       end
       
        
    end

    [~, final_y_pred_test(j)] = max(finalClassifier_test(j,:));
    
    %ccr train
    if final_y_pred_test(j) == Y_label_test(j)
        ccrtest = ccrtest + 1;
    end
    
    
end
ccrtest = ccrtest / N_test;

% final values

fprintf('\n\n Final values from multiclass SVM: \n\n')

fprintf('CCR for training set is: %.2f\n\n',ccrtrain)

fprintf('CCR for testing set is: %.2f\n\n',ccrtest)

confmat_final_train = confusionmat(final_y_pred_train, Y_label_train);

confmat_final_test = confusionmat(final_y_pred_test, Y_label_test);

fprintf('\nFinal training confusion matrix: \n')
disp(confmat_final_train);

fprintf('\nFinal testing confusion matrix: \n')
disp(confmat_final_test);

 
