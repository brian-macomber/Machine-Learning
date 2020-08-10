% EC 414 - HW 4 - Spring 2020
% DP-Means starter code

clear, clc, close all,

% ***HOW TO RUN***
% Make sure to run the section to generate the desired data
% then run the desired section

%% Generate Gaussian data:
% Add code below:
mu1 = [2 2];
mu2 = [-2 2];
mu3 = [0 -3.25];
sig1 = eye(2) * .02;
sig2 = eye(2) * .05;
sig3 = eye(2) * .07;

gm1 = gmdistribution(mu1, sig1);
gm2 = gmdistribution(mu2, sig2);
gm3 = gmdistribution(mu3, sig3);

[X1 compIdx1] = random(gm1, 50);
[X2 compIdx2] = random(gm2, 50);
[X3 compIdx3] = random(gm3, 50);

X = [X1; X2; X3];


%% Generate NBA data:
% Add code below:
NBA_data = xlsread('NBA_stats_2018_2019.xlsx');
MPG = NBA_data(:,2);
PPG = NBA_data(:,4);

NBA_cluster_data = [MPG PPG];

% HINT: readmatrix might be useful here

%% DP Means method: for gaussian data

% Parameter Initializations
lambda_vals = [.15 .4 3 20];

convergence_threshold = 1;
num_points = length(X);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%

for LAMBDA = lambda_vals

    % cluster count
    K = 1;

    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points);

    % means
    MU = [];
    MU = [MU; mean(X,1)];
    MU_previous = MU;
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;t = 0;
    k = 1;

    while (converged == 0)
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)

        k_previous = k;

        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            dists = zeros(k,1);

            for l = 1:k
                dists(l) = sqrt( (X(i,1) - MU(l,1))^2 + (X(i,2) - MU(l,2))^2 );

            end

            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            if (min(dists)^2) > LAMBDA
                    k = k + 1;
                    Z(i) = k;
                    MU(k,1) = X(i,1);
                    MU(k,2) = X(i,2);
            end      

        end

        %% CODE 3 - Form new sets of points (clusters)
        % Write code below here:
        for i = 1:num_points
           dists = zeros(k,1);

            for l = 1:k
                dists(l) = sqrt( (X(i,1) - MU(l,1))^2 + (X(i,2) - MU(l,2))^2 ); 
            end 
            index = find(dists == min(dists));
            Z(i) = index;
        end

        %% CODE 4 - Recompute means per cluster
        % Write code below here:
        for l = 1:k
                sumX1 = 0;
                sumX2 = 0;
                clusterCount = 0;
               for j = 1:length(X)
                   if Z(j) == l
                        sumX1 = sumX1 + X(j,1);
                        sumX2 = sumX2 + X(j,2);
                        clusterCount = clusterCount + 1;
                   end
               end

               %update MU here
               muNew1 = sumX1 / clusterCount;
               muNew2 = sumX2 / clusterCount;

               MU_previous(l,:) = MU(l,:);

               MU(l,1) = muNew1;
               MU(l,2) = muNew2;    
         end


        %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:
        if k == k_previous

            % finds the maximum distance that a center moved after an iteration
              maxDist = 0;
              currDist = 0;
              for l = 1:k
                  currDist = sqrt( (MU(l,1) - MU_previous(l,1))^2 + ...
                      (MU(l,2) - MU_previous(l,2))^2);
                  if currDist > maxDist
                     maxDist = currDist; 
                  end
              end

    %           maxDist = 0;
    %           for l = 1:k
    %               maxDist = maxDist + sqrt( (MU(l,1) - MU_previous(l,1))^2 + ...
    %                   (MU(l,2) - MU_previous(l,2))^2);
    %           end

              % if the max dist moved by any given center is less than the
              % threshold - convergence
             if (maxDist < convergence_threshold)
                converged = 1;
             end
        end


        %% CODE 6 - Plot final clusters after convergence 
        % Write code below here:

        if (converged)
            %%%%
            figure
            gscatter(X(:,1), X(:,2), Z)
            xlabel('X1')
            ylabel('X2')
            title(sprintf('DP means implementation for lambda = %.2f',LAMBDA))
            hold on
            plot(MU(:,1), MU(:,2), 'k*')
            hold off
        end    
    end
end




%% DP Means method: for NBA Data

% Parameter Initializations
lambda_vals = [44 100 450];

convergence_threshold = 1;
num_points = length(NBA_cluster_data);
total_indices = [1:num_points];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DP Means - Initializations for algorithm %%%

for LAMBDA = lambda_vals

    % cluster count
    K = 1;

    % sets of points that make up clusters
    L = {};
    L = [L [1:num_points]];

    % Class indicators/labels
    Z = ones(1,num_points);

    % means
    MU = [];
    MU = [MU; mean(NBA_cluster_data,1)];
    MU_previous = MU;
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initializations for algorithm:
    converged = 0;t = 0;
    k = 1;

    while (converged == 0)
        t = t + 1;
        fprintf('Current iteration: %d...\n',t)

        k_previous = k;

        %% Per Data Point:
        for i = 1:num_points

            %% CODE 1 - Calculate distance from current point to all currently existing clusters
            % Write code below here:
            dists = zeros(k,1);

            for l = 1:k
                dists(l) = sqrt( (NBA_cluster_data(i,1) - MU(l,1))^2 +...
                    (NBA_cluster_data(i,2) - MU(l,2))^2 );

            end

            %% CODE 2 - Look at how the min distance of the cluster distance list compares to LAMBDA
            % Write code below here:
            if (min(dists)^2) > LAMBDA
                    k = k + 1;
                    Z(i) = k;
                    MU(k,1) = NBA_cluster_data(i,1);
                    MU(k,2) = NBA_cluster_data(i,2);
            end      

        end

        %% CODE 3 - Form new sets of points (clusters)
        % Write code below here:
        for i = 1:num_points
           dists = zeros(k,1);

            for l = 1:k
                dists(l) = sqrt( (NBA_cluster_data(i,1) - MU(l,1))^2 +...
                    (NBA_cluster_data(i,2) - MU(l,2))^2 ); 
            end 
            index = find(dists == min(dists));
            Z(i) = index;
        end

        %% CODE 4 - Recompute means per cluster
        % Write code below here:
        for l = 1:k
                sumX1 = 0;
                sumX2 = 0;
                clusterCount = 0;
               for j = 1:length(NBA_cluster_data)
                   if Z(j) == l
                        sumX1 = sumX1 + NBA_cluster_data(j,1);
                        sumX2 = sumX2 + NBA_cluster_data(j,2);
                        clusterCount = clusterCount + 1;
                   end
               end

               %update MU here
               muNew1 = sumX1 / clusterCount;
               muNew2 = sumX2 / clusterCount;

               MU_previous(l,:) = MU(l,:);

               MU(l,1) = muNew1;
               MU(l,2) = muNew2;    
         end


        %% CODE 5 - Test for convergence: number of clusters doesn't change and means stay the same %%%
        % Write code below here:
        if k == k_previous

            % finds the maximum distance that a center moved after an iteration
              maxDist = 0;
              currDist = 0;
              for l = 1:k
                  currDist = sqrt( (MU(l,1) - MU_previous(l,1))^2 + ...
                      (MU(l,2) - MU_previous(l,2))^2);
                  if currDist > maxDist
                     maxDist = currDist; 
                  end
              end

              % if the max dist moved by any given center is less than the
              % threshold - convergence
             if (maxDist < convergence_threshold)
                converged = 1;
             end
        end


        %% CODE 6 - Plot final clusters after convergence 
        % Write code below here:

        if (converged)
            %%%%
            figure
            gscatter(NBA_cluster_data(:,1), NBA_cluster_data(:,2), Z)
            xlabel('Minutes per Game')
            ylabel('Points per Game')
            title(sprintf('DP means (NBA data) implementation for lambda = %.2f',LAMBDA))
            hold on
            plot(MU(:,1), MU(:,2), 'k*')
            hold off
        end    
    end
end



