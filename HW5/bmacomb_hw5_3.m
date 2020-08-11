%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2020
% HW 5
% Brian Macomber - U25993688
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
clear; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5.3(a)
% Generate and plot the data pointsec
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta = 0*pi/6;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,1);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% (ii)
lambda1 = 1;
lambda2 = 0.25;
theta = pi/6;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,2);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (iii)
lambda1 = 1;
lambda2 = 0.25;
theta = pi/3;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,3);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/3']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (iv)
lambda1 = 0.25;
lambda2 = 1;
theta = pi/6;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,4);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(1),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%% 5.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];
% data from (i)
lambda1 = 1;
lambda2 = 0.25;
theta = pi/6;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal_power;
    noise_power_array(i) = noise_power;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phi_maxSignal = phi_array(signal_power_array == max(signal_power_array));
phi_maxNoise = phi_array(noise_power_array == max(noise_power_array));
phi_maxSNR = phi_array(snr_array == max(snr_array));

fprintf('Phi at max signal power is %.2f\n',phi_maxSignal)
fprintf('Phi at max noise power is %.2f\n',phi_maxNoise)
fprintf('Phi at max SNR is %.2f\n',phi_maxSNR)

figure
subplot(3,1,1)
plot(phi_array, signal_power_array)
xlabel('Signal Power')
ylabel('Phi')
title('Signal Power vs. Phi')

subplot(3,1,2)
plot(phi_array, noise_power_array)
xlabel('Noise Power')
ylabel('Phi')
title('Noise Power vs. Phi')

subplot(3,1,3)
plot(phi_array, snr_array)
xlabel('SNR')
ylabel('Phi')
title('SNR vs. Phi')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
signal_noise_snr(X, Y, 0, true);
signal_noise_snr(X, Y, pi/6, true);
signal_noise_snr(X, Y, pi/3, true);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 
n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];
% data from (i)
lambda1 = 1;
lambda2 = 0.25;
theta = pi/6;

[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);


w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Xr Xc] = size(X);

X_class1 = zeros(2,50);
X_class2 = zeros(2,100);

n1 = 0;
n2 = 0;

X1 = X(:, Y==1);
X2 = X(:, Y==2);

X1_mean = mean(X1')';
X2_mean = mean(X2')';

X_mean_diff = X2_mean - X1_mean;

figure
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title('5.3c: Plotting means and wLDA over the dataset');
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

ph = quiver(X1_mean(1), X1_mean(2), X_mean_diff(1), X_mean_diff(2));
ph.LineWidth = 2;
ph.Color = 'k';

ph2 = quiver(X1_mean(1), X1_mean(2), w_LDA(1), w_LDA(2));
ph2.LineWidth = 2;
ph2.Color = 'g';

legend('Class 1', 'Class 2', 'Difference in means', 'wLDA')

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 150;

% Create CCR vs b plot

X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

maxB = b_array(ccr_array == max(ccr_array));

fprintf('the offset b that maximizes the CCr is %.2f, and the max CCR value is %.2f\n',maxB, max(ccr_array))

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plot CCR as a function of b
figure
plot(b_array, ccr_array)
xlabel('b')
ylabel('CCR')
title('CCR as a function of b')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2

eigVec1 = [cos(theta) sin(theta)]';
eigVec2 = [sin(theta) -cos(theta)]';

L = diag([lambda1, lambda2]);
V = [eigVec1, eigVec2];

covMat = V * L * (V^-1);

data1 = mvnrnd(mu1, covMat, n1);
data2 = mvnrnd(mu2, covMat, n2);

ydata1 = ones(n1, 1);
ydata2 = ones(n2, 1) + 1;

X = [data1; data2]';
Y = [ydata1; ydata2]';

end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w = [cos(phi) sin(phi)]';
[Xr Xc] = size(X);
X_phi = zeros(1, Xc);

% project X onto direction w
% for i = 1:Xc
%     X_phi(:,i) = (w / norm(w,2))' * X(:,i) * (w / norm(w,2));
% end
X_phi = w' * X;
% find empirical projected means
X_projected_phi_class1 = zeros(1,50);
X_projected_phi_class2 = zeros(1,100);

n1 = 0;
n2 = 0;
n1sum = 0;
n2sum = 0;

for i = 1:Xc
   if Y(i) == 1
       n1sum = n1sum + X_phi(i);
       n1 = n1 + 1;
       X_projected_phi_class1(n1) = X_phi(i);
   end
   if Y(i) == 2
       n2sum = n2sum + X_phi(i);
       n2 = n2 + 1;
       X_projected_phi_class2(n2) = X_phi(i);
   end
end

mu1_phi = mean(X_projected_phi_class1);
mu2_phi = mean(X_projected_phi_class2);

signal = (mu2_phi - mu1_phi)^2;

var1_phi = var(X_projected_phi_class1);
var2_phi = var(X_projected_phi_class2);

noise = ((n1/150) * var1_phi) + ((n2/150) * var2_phi);

snr = signal / noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(X_projected_phi_class1);
    plot(z1, pdf1)
    hold on;
    [pdf2,z2] = ksdensity(X_projected_phi_class2);
    plot(z2, pdf2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title(sprintf('Estimated class density estimates of data projected along phi = %.2f. Ground-truth phi = pi/6',phi))
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Xr Xc] = size(X);

X_class1 = zeros(2,50);
X_class2 = zeros(2,100);

n1 = 0;
n2 = 0;

for i = 1:Xc
   if Y(i) == 1
       n1 = n1 + 1;
       X_class1(:,n1) = X(:,i);
   end
   if Y(i) == 2
       n2 = n2 + 1;
       X_class2(:,n2) = X(:,i);
   end
end

X1_mean = mean(X_class1')';
X2_mean = mean(X_class2')';

% find covariance matrices from X

covX1 = (1 / n1) * ((X(:, Y==1) - X1_mean) * (X(:, Y==1) - X1_mean)');
covX2 = (1 / n2) * ((X(:, Y==2) - X2_mean) * (X(:, Y==2) - X2_mean)');

Sx_avg = ((n1 / Xc) * covX1 ) + ((n2 / Xc) * covX2);

w_LDA = (Sx_avg)^-1 * (X2_mean - X1_mean);

end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[Xr Xc] = size(X);

Y_test = ones(1, Xc);

for i = 1:Xc
    
    decision = w_LDA' * X(:,i) + b;
    
    if decision <= 0
        Y_test(i) = 1;
    else
        Y_test(i) = 2;
    end
    
end

correctY = 0;

for i = 1:Xc
    if Y(i) == Y_test(i)
        correctY = correctY + 1;
    end
end

ccr = correctY / Xc;


end