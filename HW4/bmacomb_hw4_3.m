% EC414 - HW4 - question 3


clear,clc
%% generate gaussian data

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

x1 = X(:,1);
x2 = X(:,2);
x1min = min(x1);
x1max = max(x1);
x2min = min(x2);
x2max = max(x2);

minRange = min([x1min x2min]);
maxRange = max([x1max x2max]);

%% run k-means

% run k means on 10 random implementations for each value of k:

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
f_k_lambda = zeros(1,9);

for lambda = 15:5:30
    
    for i = 1:9
       f_k_lambda(i) = WCSS_for_k(i) + (lambda * (i+1));
    end
    figure
    plot(k,f_k_lambda)
    xlabel('k')
    ylabel('f(k,lambda)')
    title(sprintf('Selecting k-means with penalty lambda = %d',lambda))
    
end

