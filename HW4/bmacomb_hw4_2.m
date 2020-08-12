% EC 414 - HW 4 - Spring 2020

% K-Means starter code

% ****HOW TO RUN CODE*****
% Depending on the part being teseted, load the appropriate data by only
% running one of the sections below
% Then run the desired section

% Alternative, run the entire file and it will still produce all the
% results

clear, clc, close all,

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

colors = zeros(150,3);
colors(1:50,1) = 1;
colors(51:100,2) = 1;
colors(101:150,3) = 1;

X = [X1; X2; X3];

figure
scatter(X(:,1), X(:,2),50,colors,'.')
xlabel('x1')
ylabel('x2')
title('3 two dimensional Gaussian clusters')






%% Generate NBA data:
% Add code below:

% HINT: readmatrix might be useful here

NBA_data = xlsread('NBA_stats_2018_2019.xlsx');
MPG = NBA_data(:,2);
PPG = NBA_data(:,4);

plot(MPG, PPG, 'b.')
xlabel('Minutes per game')
ylabel('Points per game')
title('2018-2019 NBA stats for MPG and PPG')

% Problem 4.2(f): Generate Concentric Rings Dataset using
% sample_circle.m provided to you in the HW 4 folder on Blackboard.
%% Concentric circle data

[conCircle_data circle_labels] = sample_circle(3,500);
gscatter(conCircle_data(:,1), conCircle_data(:,2), circle_labels)
xlabel('X1')
ylabel('X2')
title('Dataset with 3 concentric circles of 500 data points each')

%% 4.2a:  K-Means implementation
% Add code below

K = 3;
MU_init = [3 3;-4 -1;2 -4];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for j = 1:length(X)
        dists = zeros(K,1);
        
        for l = 1:K
            % calc distance between current point and each center
            dists(l) = sqrt( (X(j,1) - MU_current(l,1))^2 + (X(j,2) - MU_current(l,2))^2 );
        end
        % assign index to closest 
        clusterIndex = find(dists == min(dists));
        labels(j) = clusterIndex;
    end
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for l = 1:K
        sumX1 = 0;
        sumX2 = 0;
        clusterCount = 0;
       for j = 1:length(X)
           if labels(j) == l
                sumX1 = sumX1 + X(j,1);
                sumX2 = sumX2 + X(j,2);
                clusterCount = clusterCount + 1;
           end
       end
       
       %update MU here
       muNew1 = sumX1 / clusterCount;
       muNew2 = sumX2 / clusterCount;
       
       MU_previous(l,:) = MU_current(l,:);
       
       MU_current(l,1) = muNew1;
       MU_current(l,2) = muNew2;    
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    % finds the maximum distance that a center moved after an iteration
      maxDist = 0;
      for l = 1:K
          maxDist = maxDist + sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
              (MU_current(l,2) - MU_previous(l,2))^2);
      end
    
      % if the max dist moved by any given center is less than the
      % threshold - convergence
     if (maxDist < convergence_threshold)
        converged = 1;
     end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure
        gscatter(X(:,1), X(:,2), labels);
        
         hold on
        
        plot(MU_current(:,1), MU_current(:,2), 'k*')
        
        hold off
        xlabel('X1')
        ylabel('X2')
        title('4.2a: k-means using k = 3 on gaussian dataset')
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        WCSS = 0;
        for j = 1:length(X)
            WCSS = WCSS + ( sqrt( (X(j,1) - MU_current(labels(j),1))^2 + (X(j,2) - MU_current(labels(j),2))^2 ) )^2;
            
        end
        fprintf('WCSS of 4.2a is %.3f. \n',WCSS)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%% 4.2b 

% K-Means implementation using different initialization

K = 3;
MU_init = [-.14 2.61;3.15 -.84;-3.28 -.84];

MU_previous = MU_init;
MU_current = MU_init;

% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for j = 1:length(X)
        dists = zeros(K,1);
        
        for l = 1:K
            % calc distance between current point and each center
            dists(l) = sqrt( (X(j,1) - MU_current(l,1))^2 + (X(j,2) - MU_current(l,2))^2 );
        end
        % assign index to closest 
        clusterIndex = find(dists == min(dists));
        labels(j) = clusterIndex;
    end
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for l = 1:K
        sumX1 = 0;
        sumX2 = 0;
        clusterCount = 0;
       for j = 1:length(X)
           if labels(j) == l
                sumX1 = sumX1 + X(j,1);
                sumX2 = sumX2 + X(j,2);
                clusterCount = clusterCount + 1;
           end
       end
       
       %update MU here
       muNew1 = sumX1 / clusterCount;
       muNew2 = sumX2 / clusterCount;
       
       MU_previous(l,:) = MU_current(l,:);
       
       MU_current(l,1) = muNew1;
       MU_current(l,2) = muNew2;    
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    % finds the maximum distance that a center moved after an iteration
      maxDist = 0;
      for l = 1:K
          maxDist = maxDist + sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
              (MU_current(l,2) - MU_previous(l,2))^2);
      end
    
      % if the max dist moved by any given center is less than the
      % threshold - convergence
     if (maxDist < convergence_threshold)
        converged = 1;
     end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure
        gscatter(X(:,1), X(:,2), labels);
        
         hold on
        
        plot(MU_current(:,1), MU_current(:,2), 'k*')
        
        hold off
        xlabel('X1')
        ylabel('X2')
        title('4.2b: k-means using k = 3 on gaussian dataset')
        
       
        
        %% If converged, get WCSS metric
        % Add code below
        WCSS = 0;
        for j = 1:length(X)
            WCSS = WCSS + ( sqrt( (X(j,1) - MU_current(labels(j),1))^2 + (X(j,2) - MU_current(labels(j),2))^2 ) )^2;
            
        end
        fprintf('WCSS of 4.2b is %.3f. \n',WCSS)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% 4.2c

% find range to generate random intitializations for
x1 = X(:,1);
x2 = X(:,2);
x1min = min(x1);
x1max = max(x1);
x2min = min(x2);
x2max = max(x2);

minRange = min([x1min x2min]);
maxRange = max([x1max x2max]);

bestWCSS = inf;
WCSSindex = 0;



% K-Means implementation

% Add code below
for i = 1:10
    K = 3;
    MU_init = (maxRange - minRange)*rand(K,2) + minRange;

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    labels = ones(length(X),1);
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        for j = 1:length(X)
            dists = zeros(K,1);

            for l = 1:K
                % calc distance between current point and each center
                dists(l) = sqrt( (X(j,1) - MU_current(l,1))^2 + (X(j,2) - MU_current(l,2))^2 );
            end
            % assign index to closest 
            clusterIndex = find(dists == min(dists));
            labels(j) = clusterIndex;
        end

        %% CODE - Mean Updating - Update the cluster means
        % Write code below here:
        for l = 1:K
            sumX1 = 0;
            sumX2 = 0;
            clusterCount = 0;
           for j = 1:length(X)
               if labels(j) == l
                    sumX1 = sumX1 + X(j,1);
                    sumX2 = sumX2 + X(j,2);
                    clusterCount = clusterCount + 1;
               end
           end

           %update MU here
           muNew1 = sumX1 / clusterCount;
           muNew2 = sumX2 / clusterCount;

           MU_previous(l,:) = MU_current(l,:);

           MU_current(l,1) = muNew1;
           MU_current(l,2) = muNew2;    
        end

        %% CODE 4 - Check for convergence 
        % Write code below here:
        % finds the maximum distance that a center moved after an iteration
          maxDist = 0;
          currDist = 0;
          for l = 1:K
              currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
                  (MU_current(l,2) - MU_previous(l,2))^2);
              if currDist > maxDist
                 maxDist = currDist; 
              end
          end
            
          % if the max dist moved by any given center is less than the
          % threshold - convergence
         if (maxDist < convergence_threshold)
            converged = 1;
         end

        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            
            % don't plot here, wait until after to plot
                
            %% If converged, get WCSS metric
            % Add code below
            WCSS = 0;
            for j = 1:length(X)
                WCSS = WCSS + ( sqrt( (X(j,1) - MU_current(labels(j),1))^2 + (X(j,2) - MU_current(labels(j),2))^2 ) )^2;

            end
            fprintf('WCSS of 4.2b of trial %d is %.3f. \n\n',i,WCSS)
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    if WCSS < bestWCSS
        bestWCSS = WCSS;
        WCSSindex = i;
        MU_best = MU_current;
    end
    
    
end

fprintf('The lowest WCSS value was during trial %d.\n',WCSSindex)

% now run k-means using MU_best as the centers

MU_previous = MU_best;
MU_current = MU_best;

% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for j = 1:length(X)
        dists = zeros(K,1);
        
        for l = 1:K
            % calc distance between current point and each center
            dists(l) = sqrt( (X(j,1) - MU_current(l,1))^2 + (X(j,2) - MU_current(l,2))^2 );
        end
        % assign index to closest 
        clusterIndex = find(dists == min(dists));
        labels(j) = clusterIndex;
    end
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for l = 1:K
        sumX1 = 0;
        sumX2 = 0;
        clusterCount = 0;
       for j = 1:length(X)
           if labels(j) == l
                sumX1 = sumX1 + X(j,1);
                sumX2 = sumX2 + X(j,2);
                clusterCount = clusterCount + 1;
           end
       end
       
       %update MU here
       muNew1 = sumX1 / clusterCount;
       muNew2 = sumX2 / clusterCount;
       
       MU_previous(l,:) = MU_current(l,:);
       
       MU_current(l,1) = muNew1;
       MU_current(l,2) = muNew2;    
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    % finds the maximum distance that a center moved after an iteration
      maxDist = 0;
      currDist = 0;
      for l = 1:K
          currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
              (MU_current(l,2) - MU_previous(l,2))^2);
          if currDist > maxDist
             maxDist = currDist; 
          end
      end

      % if the max dist moved by any given center is less than the
      % threshold - convergence
     if (maxDist < convergence_threshold)
        converged = 1;
     end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure
        gscatter(X(:,1), X(:,2), labels);
        
         hold on
        
        plot(MU_current(:,1), MU_current(:,2), 'k*')
        
        hold off
        xlabel('X1')
        ylabel('X2')
        title('4.2c: best WCSS for k-means using k = 3 on gaussian dataset after 10 random trials')
        
       
        
        %% If converged, get WCSS metric
        
        % used saved best WCSS from before
        fprintf('Best WCSS of 4.2c is %.3f. \n',bestWCSS)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%% 4.2d 

% run k means on 10 random implementations for each value of k:

x1 = X(:,1);
x2 = X(:,2);
x1min = min(x1);
x1max = max(x1);
x2min = min(x2);
x2max = max(x2);

minRange = min([x1min x2min]);
maxRange = max([x1max x2max]);

bestWCSS = inf;
WCSSindex = 0;


% initializations
labels = ones(length(X),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

WCSS_for_k = zeros(1,9);

for k = 2:10
    bestWCSS = inf;
    
   for i = 1:10
    K = k;
    MU_init = (maxRange - minRange)*rand(K,2) + minRange;

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    labels = ones(length(X),1);
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        for j = 1:length(X)
            dists = zeros(K,1);

            for l = 1:K
                % calc distance between current point and each center
                dists(l) = sqrt( (X(j,1) - MU_current(l,1))^2 + (X(j,2) - MU_current(l,2))^2 );
            end
            % assign index to closest 
            clusterIndex = find(dists == min(dists));
            labels(j) = clusterIndex;
        end

        %% CODE - Mean Updating - Update the cluster means
        % Write code below here:
        for l = 1:K
            sumX1 = 0;
            sumX2 = 0;
            clusterCount = 0;
           for j = 1:length(X)
               if labels(j) == l
                    sumX1 = sumX1 + X(j,1);
                    sumX2 = sumX2 + X(j,2);
                    clusterCount = clusterCount + 1;
               end
           end

           %update MU here
           muNew1 = sumX1 / clusterCount;
           muNew2 = sumX2 / clusterCount;

           MU_previous(l,:) = MU_current(l,:);

           MU_current(l,1) = muNew1;
           MU_current(l,2) = muNew2;    
        end

        %% CODE 4 - Check for convergence 
        % Write code below here:
        % finds the maximum distance that a center moved after an iteration
          maxDist = 0;
          currDist = 0;
          for l = 1:K
              currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
                  (MU_current(l,2) - MU_previous(l,2))^2);
              if currDist > maxDist
                 maxDist = currDist; 
              end
          end

          % if the max dist moved by any given center is less than the
          % threshold - convergence
         if (maxDist < convergence_threshold)
            converged = 1;
         end

        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')

            %% If converged, get WCSS metric
            % Add code below
            WCSS = 0;
            for j = 1:length(X)
                WCSS = WCSS + ( sqrt( (X(j,1) - MU_current(labels(j),1))^2 + (X(j,2) - MU_current(labels(j),2))^2 ) )^2;

            end
            fprintf('WCSS of 4.2b of trial %d for k = %d is %.3f. \n\n',i,k,WCSS)
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    if WCSS < bestWCSS
        bestWCSS = WCSS;
    end
    
    
   end
   
   WCSS_for_k(k-1) = bestWCSS;
    
    
    
end

% now plot the best WCSS values vs the k values
k = 2:10;

plot(k,WCSS_for_k,'k')
xlabel('k')
ylabel('best WCSS values out of 10 trials')
title('k vs. WCSS for the best out of 10 trials')

%% 4.2e - k-means with k=10 on the NBA data

bestWCSS = inf;
WCSSindex = 0;

NBA_cluster_data = [MPG PPG];

x1 = NBA_cluster_data(:,1);
x2 = NBA_cluster_data(:,2);
x1min = min(x1);
x1max = max(x1);
x2min = min(x2);
x2max = max(x2);

minRange = min([x1min x2min]);
maxRange = max([x1max x2max]);

% K-Means implementation

% Add code below
for i = 1:10
    K = 10;
    MU_init = (maxRange - minRange)*rand(K,2) + minRange;

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    labels = ones(length(NBA_cluster_data),1);
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        for j = 1:length(NBA_cluster_data)
            dists = zeros(K,1);

            for l = 1:K
                % calc distance between current point and each center
                dists(l) = sqrt( (NBA_cluster_data(j,1) - MU_current(l,1))^2 ...
                    + (NBA_cluster_data(j,2) - MU_current(l,2))^2 );
            end
            % assign index to closest 
            clusterIndex = find(dists == min(dists));
            labels(j) = clusterIndex;
        end

        %% CODE - Mean Updating - Update the cluster means
        % Write code below here:
        for l = 1:K
            sumX1 = 0;
            sumX2 = 0;
            clusterCount = 0;
           for j = 1:length(NBA_cluster_data)
               if labels(j) == l
                    sumX1 = sumX1 + NBA_cluster_data(j,1);
                    sumX2 = sumX2 + NBA_cluster_data(j,2);
                    clusterCount = clusterCount + 1;
               end
           end

           %update MU here
           muNew1 = sumX1 / clusterCount;
           muNew2 = sumX2 / clusterCount;

           MU_previous(l,:) = MU_current(l,:);

           MU_current(l,1) = muNew1;
           MU_current(l,2) = muNew2;    
        end

        %% CODE 4 - Check for convergence 
        % Write code below here:
        % finds the maximum distance that a center moved after an iteration
          maxDist = 0;
          currDist = 0;
          for l = 1:K
              currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
                  (MU_current(l,2) - MU_previous(l,2))^2);
              if currDist > maxDist
                 maxDist = currDist; 
              end
          end

          % if the max dist moved by any given center is less than the
          % threshold - convergence
         if (maxDist < convergence_threshold)
            converged = 1;
         end

        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            
            % don't plot here, wait until after to plot
                
            %% If converged, get WCSS metric
            % Add code below
            WCSS = 0;
            for j = 1:length(NBA_cluster_data)
                WCSS = WCSS + ( sqrt( (NBA_cluster_data(j,1) - MU_current(labels(j),1))^2 +...
                    (NBA_cluster_data(j,2) - MU_current(labels(j),2))^2 ) )^2;

            end
            fprintf('WCSS of 4.2b of trial %d is %.3f. \n\n',i,WCSS)
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    if WCSS < bestWCSS
        bestWCSS = WCSS;
        WCSSindex = i;
        MU_best = MU_current;
    end
    
    
end

fprintf('The lowest WCSS value was during trial %d.\n',WCSSindex)

% now run k-means using MU_best as the centers

MU_previous = MU_best;
MU_current = MU_best;

% initializations
labels = ones(length(NBA_cluster_data),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for j = 1:length(NBA_cluster_data)
        dists = zeros(K,1);
        
        for l = 1:K
            % calc distance between current point and each center
            dists(l) = sqrt( (NBA_cluster_data(j,1) - MU_current(l,1))^2 + ...
                (NBA_cluster_data(j,2) - MU_current(l,2))^2 );
        end
        % assign index to closest 
        clusterIndex = find(dists == min(dists));
        labels(j) = clusterIndex;
    end
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for l = 1:K
        sumX1 = 0;
        sumX2 = 0;
        clusterCount = 0;
       for j = 1:length(NBA_cluster_data)
           if labels(j) == l
                sumX1 = sumX1 + NBA_cluster_data(j,1);
                sumX2 = sumX2 + NBA_cluster_data(j,2);
                clusterCount = clusterCount + 1;
           end
       end
       
       %update MU here
       muNew1 = sumX1 / clusterCount;
       muNew2 = sumX2 / clusterCount;
       
       MU_previous(l,:) = MU_current(l,:);
       
       MU_current(l,1) = muNew1;
       MU_current(l,2) = muNew2;    
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    % finds the maximum distance that a center moved after an iteration
      maxDist = 0;
      currDist = 0;
      for l = 1:K
          currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
              (MU_current(l,2) - MU_previous(l,2))^2);
          if currDist > maxDist
             maxDist = currDist; 
          end
      end
    
      % if the max dist moved by any given center is less than the
      % threshold - convergence
     if (maxDist < convergence_threshold)
        converged = 1;
     end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure
        gscatter(NBA_cluster_data(:,1), NBA_cluster_data(:,2), labels);
        
         hold on
        
        plot(MU_current(:,1), MU_current(:,2), 'k*')
        
        hold off
        xlabel('X1: Minutes per game')
        ylabel('X2: Points per game')
        title('4.2e: k-means on NBA dataset')
        
       
        
        %% If converged, get WCSS metric

    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% 4.2f - failure of k means on the concentric data set

% find range to generate random intitializations for
x1 = conCircle_data(:,1);
x2 = conCircle_data(:,2);
x1min = min(x1);
x1max = max(x1);
x2min = min(x2);
x2max = max(x2);

minRange = min([x1min x2min]);
maxRange = max([x1max x2max]);

bestWCSS = inf;
WCSSindex = 0;



% K-Means implementation

% Add code below
for i = 1:10
    K = 3;
    MU_init = (maxRange - minRange)*rand(K,2) + minRange;

    MU_previous = MU_init;
    MU_current = MU_init;

    % initializations
    labels = ones(length(conCircle_data),1);
    converged = 0;
    iteration = 0;
    convergence_threshold = 0.025;

    while (converged==0)
        iteration = iteration + 1;
        fprintf('Iteration: %d\n',iteration)

        %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
        % Write code below here:
        for j = 1:length(conCircle_data)
            dists = zeros(K,1);

            for l = 1:K
                % calc distance between current point and each center
                dists(l) = sqrt( (conCircle_data(j,1) - MU_current(l,1))^2 +...
                    (conCircle_data(j,2) - MU_current(l,2))^2 );
            end
            % assign index to closest 
            clusterIndex = find(dists == min(dists));
            labels(j) = clusterIndex;
        end

        %% CODE - Mean Updating - Update the cluster means
        % Write code below here:
        for l = 1:K
            sumX1 = 0;
            sumX2 = 0;
            clusterCount = 0;
           for j = 1:length(conCircle_data)
               if labels(j) == l
                    sumX1 = sumX1 + conCircle_data(j,1);
                    sumX2 = sumX2 + conCircle_data(j,2);
                    clusterCount = clusterCount + 1;
               end
           end

           %update MU here
           muNew1 = sumX1 / clusterCount;
           muNew2 = sumX2 / clusterCount;

           MU_previous(l,:) = MU_current(l,:);

           MU_current(l,1) = muNew1;
           MU_current(l,2) = muNew2;    
        end

        %% CODE 4 - Check for convergence 
        % Write code below here:
        % finds the maximum distance that a center moved after an iteration
          maxDist = 0;
          currDist = 0;
          for l = 1:K
              currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
                  (MU_current(l,2) - MU_previous(l,2))^2);
              if currDist > maxDist
                 maxDist = currDist; 
              end
          end

          % if the max dist moved by any given center is less than the
          % threshold - convergence
         if (maxDist < convergence_threshold)
            converged = 1;
         end

        %% CODE 5 - Plot clustering results if converged:
        % Write code below here:
        if (converged == 1)
            fprintf('\nConverged.\n')
            
            % don't plot here, wait until after to plot
                
            %% If converged, get WCSS metric
            % Add code below
            WCSS = 0;
            for j = 1:length(conCircle_data)
                WCSS = WCSS + ( sqrt( (conCircle_data(j,1) - MU_current(labels(j),1))^2 ...
                    + (conCircle_data(j,2) - MU_current(labels(j),2))^2 ) )^2;

            end
            fprintf('WCSS of 4.2b of trial %d is %.3f. \n\n',i,WCSS)
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    if WCSS < bestWCSS
        bestWCSS = WCSS;
        WCSSindex = i;
        MU_best = MU_current;
    end
    
    
end

fprintf('The lowest WCSS value was during trial %d.\n',WCSSindex)

% now run k-means using MU_best as the centers

MU_previous = MU_best;
MU_current = MU_best;

% initializations
labels = ones(length(conCircle_data),1);
converged = 0;
iteration = 0;
convergence_threshold = 0.025;

while (converged==0)
    iteration = iteration + 1;
    fprintf('Iteration: %d\n',iteration)

    %% CODE - Assignment Step - Assign each data observation to the cluster with the nearest mean:
    % Write code below here:
    for j = 1:length(conCircle_data)
        dists = zeros(K,1);
        
        for l = 1:K
            % calc distance between current point and each center
            dists(l) = sqrt( (conCircle_data(j,1) - MU_current(l,1))^2 +...
                (conCircle_data(j,2) - MU_current(l,2))^2 );
        end
        % assign index to closest 
        clusterIndex = find(dists == min(dists));
        labels(j) = clusterIndex;
    end
    
    %% CODE - Mean Updating - Update the cluster means
    % Write code below here:
    for l = 1:K
        sumX1 = 0;
        sumX2 = 0;
        clusterCount = 0;
       for j = 1:length(conCircle_data)
           if labels(j) == l
                sumX1 = sumX1 + conCircle_data(j,1);
                sumX2 = sumX2 + conCircle_data(j,2);
                clusterCount = clusterCount + 1;
           end
       end
       
       %update MU here
       muNew1 = sumX1 / clusterCount;
       muNew2 = sumX2 / clusterCount;
       
       MU_previous(l,:) = MU_current(l,:);
       
       MU_current(l,1) = muNew1;
       MU_current(l,2) = muNew2;    
    end
    
    %% CODE 4 - Check for convergence 
    % Write code below here:
    % finds the maximum distance that a center moved after an iteration
      maxDist = 0;
      currDist = 0;
      for l = 1:K
          currDist = sqrt( (MU_current(l,1) - MU_previous(l,1))^2 + ...
              (MU_current(l,2) - MU_previous(l,2))^2);
          if currDist > maxDist
             maxDist = currDist; 
          end
      end
    
      % if the max dist moved by any given center is less than the
      % threshold - convergence
     if (maxDist < convergence_threshold)
        converged = 1;
     end
    
    %% CODE 5 - Plot clustering results if converged:
    % Write code below here:
    if (converged == 1)
        fprintf('\nConverged.\n')
        figure
        gscatter(conCircle_data(:,1), conCircle_data(:,2), labels);
        
         hold on
        
        plot(MU_current(:,1), MU_current(:,2), 'k*')
        
        hold off
        xlabel('X1')
        ylabel('X2')
        title('4.2f: k-means on concentric circle data with k=3')
        
       
        
        %% If converged, get WCSS metric
        
        % used saved best WCSS from before
        fprintf('Best WCSS of 4.2f is %.3f. \n',bestWCSS)
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
