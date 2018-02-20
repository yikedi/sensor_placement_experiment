% Find the near-optimal place under Mutual Information criteria for sampling of a GP with a known Covariance
clear all; close all; clc;
%% Define the 2-d mesh (space): X,Y
[X,Y] = meshgrid(0.1:0.1:1,0.1:0.1:1); % X: n*n dimentional mesh; Y: n*n dimentional mesh
N = prod(size(X)); % mesh scale, N = n*n, represents the total number of mesh cells
XX = reshape(X,N,1); % translate a n*n matrix to a N*1 vector, elements are x-axis coordinates
YY = reshape(Y,N,1); % translate a n*n matrix to a N*1 vector, elements are y-axis coordinates
mesh = [XX YY]; % 2-D mesh
%% Define the kernel paramters of interest
% use an RBF covariance function: k(x_i, x_j) = alpha exp{-gamma(x_n-x_m)^2}
alpha = 10;
gamma = 0.1;
%% Create the covariance function
Sigma = zeros(N,N);
for n = 1:N
    for m = 1:N
        Sigma(n,m) = alpha*exp(-gamma*sum((mesh(n,:) - mesh(m,:)).^2)); % Kernel
    end
end
% add some jitter for numerical stability
Sigma = Sigma + 1e-6*eye(N);
%% Sample from a 2-d GP prior
f = mvnrnd(repmat(0,N,1),Sigma,1); % 2-D Gaussian distribution
%% Plot the Gaussian random field
figure()
h=surf(X,Y,reshape(f,size(X))); view(3); 
drawnow;
%% Find the near-optimal place
V_sigma = 1:size(Sigma,1);
F_mi = sfo_fn_mi(Sigma,V_sigma);
k = 3; % Number of the sampling locations
opt.greedy_initial_sset=1:5
[A,scores,evals] = sfo_greedy_lazy(F_mi,V_sigma,k,opt); 
A
%% Plot the selected sampling places
figure()
plot(mesh(A,1),mesh(A,2),'bs','markerfacecolor','blue'); 
xlim([0 1.1]); ylim([0 1.1]); grid on;
set(gca,'xtick',[0:0.1:1.1]); set(gca,'ytick',[0:0.1:1.1]);