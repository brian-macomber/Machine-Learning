function [lambda5, k_] = skeleton_hw6_2()
%% Q6.2  Brian Macomber - U25993688
%% Load AT&T Cambridge, Face images data set
    img_size = [112,92];   % image size (rows,columns)
    % Load the ATT Face data set using load_faces()
    
    % each row is a picture
    X = load_faces();
    
    %% Compute mean face and the Auto Covariance Matrix
    % compute X_tilde
    [Xr, Xc] = size(X);
    X_mean = mean(X);
    
    
    % Compute covariance using X_tilde
    X_tilde = X - X_mean;
    
    S_x = (1 / Xr ) * (X_tilde') * X_tilde;
    
    %% Find Eigen Value Decomposition of auto covariance matrix
    [U, LAMBDA] = eig(S_x);
    
    
    %% Sort eigen values and corresponding eigen vectors and obtain U, Lambda
    [n, ~] = size(U);
    
    sortedEigs = zeros(n, 1);
    sortedU = zeros(n, n);
    sortedLAMBDA = zeros(n, n);
    
    eigValues = sum(LAMBDA);
 
    for i = 1:n
        sortedEigs(i) = max(eigValues);
        index = find(eigValues == max(eigValues));
        
        sortedU(:, i) = U(:, index(1));
        sortedLAMBDA(:, i) = LAMBDA(:, index(1));
        
        eigValues(index(1)) = -1;
    end
    
    %% Find principle components: Y
    %%%%% TODO
    Y = sortedU' * (X_tilde');
    
%% Q6.2 a) Visualize loaded images and mean face
    disp('********** 6.2a **********')
    figure(1)
    sgtitle('Data Visualization')
    
    % Visualize image 120 in the dataset
    % practise using subplots for later parts
    subplot(1,2,1)

    
    imshow(uint8(reshape(X(120,:), img_size)));
    title('image 120')
    % Visualize the Average face image
    subplot(1,2,2)

    imshow(uint8(reshape(X_mean, img_size)));
    title('mean image')
%% Q6.2 b) Analysing computed eigen values
    disp('********** 6.2b **********')
    warning('off')
    
    % Report first 5 eigen values
    lambda5 = sortedEigs(1:5)';
    
    for i = 1:length(lambda5)
       fprintf('eigenvalue %d: %.3f\n',i,lambda5(i)) 
    end
    
    % Plot trends in Eigen values and k
    k = 1:450;
    
    figure(2)
    sgtitle('Eigen Value trends w.r.t k')

    % Plot the eigen values w.r.t to k
    subplot(1,2,1)

    plot(k, sortedEigs(1:length(k)))
    xlabel('k'); ylabel('eigenvalues');
    
    % Plot running sum of eigen vals over total sum of eigen values w.r.t k

    eigenFracs = zeros(1,n);
    
    for i = 1:n
       for j = 1:i
           eigenFracs(i) = eigenFracs(i) + sortedEigs(j);
       end
       % round each to 2 decimal points
       eigenFracs(i) = round(eigenFracs(i) / sum(sortedEigs), 2);
    end
    
    
    subplot(1,2,2)

    plot(k, eigenFracs(1:length(k)));
    xlabel('k'); ylabel('eigen fractions');
    
    
    % find & report k for which Eig fraction = [0.51, 0.75, 0.9, 0.95, 0.99]
    ef = [0.51, 0.75, 0.9, 0.95, 0.99];
    k_ = zeros(1, length(ef));
    
    for i = 1:length(ef)
       index = find(eigenFracs == ef(i));
       k_(i) = k(index(1));
       fprintf('k value for eigFraction = %.2f is %d\n', ef(i), k_(i))
    end
    
%% Q6.2 c) Approximating an image using eigen faces
    disp('********** 6.2c **********')
    test_img_idx = 43;
    test_img = X(test_img_idx,:); 
    
    % Computing eigen face coefficients
    
    K = [0,1,2,k_];

    figure(3)
    sgtitle('Approximating original image by adding eigen faces')
    
    
    for i = 1:length(K) + 1
        X_hat = zeros(1,n);
        
        if i == 1   % mean face condition
            subplot(3,3,i)
            imshow(uint8(reshape(X_mean, img_size)));
            title('mean face')
        else
            if i == length(K) + 1   % original face condition
                subplot(3,3,i)
                imshow(uint8(reshape(test_img, img_size)));
                title('Original Image')

            else

                X_hat = X_mean' + sortedU(:,1:K(i)) * Y(1:K(i),43);

                % plot actual thing here
                subplot(3,3,i)
                imshow(uint8(reshape(X_hat, img_size)));
                title(sprintf('image for k = %d',K(i)))
            end
        end
        
    end

%% Q6.2 d) Principle components and corresponding properties in images
    clear;
    clc;
    
    disp('********** 6.2d **********')
% Loading and pre-processing MNIST Data-set
    % Data Prameters
    q = 5;                  % number of quantile points
    noi = 3;                % Number of interest
    img_size = [16, 16];
    
    % load mnist into workspace
    mnist = load('mnist256.mat').mnist;
    label = mnist(:,1);
    X = mnist(:,(2:end));
    num_idx = (label == noi);
    X = X(num_idx,:);
    [n,~] = size(X);
    
    %% Compute mean face and the Auto Covariance Matrix
    % compute X_tilde
    X_mean = mean(X);
    
    X_tilde = X - X_mean;
    
    % Compute covariance using X_tilde
    
    S_x = (1 / n) * (X_tilde') * X_tilde; 
    
    %% Find Eigen Value Decomposition
    [U, LAMBDA] = eig(S_x);
    
    %% Sort eigen values and corresponding eigen vectors
    [n, ~] = size(U);
    
    sortedEigs = zeros(n, 1);
    sortedU = zeros(n, n);
    sortedLAMBDA = zeros(n, n);
    
    eigValues = sum(LAMBDA);
 
    for i = 1:n
        sortedEigs(i) = max(eigValues);
        index = find(eigValues == max(eigValues));
        
        sortedU(:, i) = U(:, index(1));
        sortedLAMBDA(:, i) = LAMBDA(:, index(1));
        
        eigValues(index(1)) = -1;
    end
    
    
    %% Find principle components
    Y = sortedU' * (X_tilde');
    
    %% Computing first 2 priciple components
    Y_1 = (sortedU(:,1)' * X_tilde')' ;
    Y_2 = (sortedU(:,2)' * X_tilde')' ;
    
  

    % finding quantile points
    quantile_vals = [0.05, .25, .5, .75, .95] * 100;
    %   (Hint: Use the provided fucntion - quantile_points())
    
    Y_1_percentiles = percentile_values(Y_1, quantile_vals);
    Y_2_percentiles = percentile_values(Y_2, quantile_vals);
    
    
    % Finding the cartesian product of quantile points to find grid corners
    [Xgrid, Ygrid] = ndgrid(Y_1_percentiles, Y_2_percentiles);
    
    % get all of the intersection points
    cartProd = [Xgrid(:) Ygrid(:)];
    
    
    %% find closest coordinates to grid corner coordinates    
    %% and  Find the actual images with closest coordinate index    
    
    % find all distances between the intersections and all y1, y2 pairs
    Y_1_2 = [Y_1 Y_2];
        
    D = pdist2(cartProd, Y_1_2, 'euclidean');
    
    [minDists, min_indices] = min(D');
    
    closestPoints = zeros(25,2);
    
    for i = 1:length(min_indices)     
        closestPoints(i,1) = Y_1_2(min_indices(i),1);   
        closestPoints(i,2) = Y_1_2(min_indices(i),2);
    end
    

    %% Visualize loaded images
    % random image in dataset
    figure(4)
    sgtitle('Data Visualization')

    % Visualize the 120th image
    subplot(1,2,1)
    imshow((reshape(X(120,:), img_size)));
    title('120th image in dataset')
    
    % Average face image
    subplot(1,2,2)
    imshow((reshape(X_mean, img_size)));
    title('Mean')

    
    %% Image Projections on Principle components and the corresponding features
    
    figure(5)    
    hold on
    grid on
    
    % Plotting the Principle component 1 vs 2, Principle component. Draw the
    % grid formed by the quantile points and highlight points closest to the 
    % quantile grid corners
    scatter(Y_1, Y_2)
    
    scatter(closestPoints(:,1), closestPoints(:,2), 'r', 'filled')


    xlabel('Principle component 1')
    ylabel('Principle component 2')
    title('Closest points to quantile grid corners')
    
    xticks(Y_1_percentiles)
    yticks(Y_2_percentiles)
    
    hold off
    
    figure(6)
    sgtitle('Images at corresponding red dots')
    hold on
    % Plot the images corresponding to points closest to the quantile grid 
    % corners. Use subplot to put all images in a single figure in a grid
    
    for i = 1:25
       subplot(5,5,i)
       
       imshow((reshape(X(min_indices(i), :), img_size)));
       title(sprintf('Red dot at index %d',min_indices(i)))
        
    end
    
    
    hold off    
end