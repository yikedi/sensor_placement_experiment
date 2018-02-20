
clear all;
close all;

%get_y= @(x) x(:,1).*exp(-x(:,1).^2-x(:,2).^2);
get_y=@(x) x(:,1)+x(:,2);
% set up the grid
[X1 X2]=meshgrid(-5:0.2:5,-5:0.2:5);
x1_grid=reshape(X1,[],1);
x2_grid=reshape(X2,[],1);
grid=[x1_grid,x2_grid];
ytrue=get_y(grid);

% set up starting training points by randomly select n points from the grid
% Also randomly select n+k points for later comparison

radius=1;
n=10;
k=10;
index=randsample(length(grid),n);
xtrain1=zeros(n+k,2);
ytrain1=zeros(n+k,1);
xtrain1(1:n,:)=grid(index,:); 
ytrain1(1:n,:)=get_y(xtrain1(1:n,:));


% get the current gaussian 


for l=1:k
    
gprMdl=fitrgp(xtrain1(1:n+l-1,:),ytrain1(1:n+l-1,:),'kernelFunction','squaredexponential');    
% gprMdl=fitrgp(xtrain1,ytrain1,'FitMethod','exact','PredictMethod','exact',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,...
%     'MaxObjectiveEvaluations',10));

last_point=xtrain1(n+l-1,:);
n_division=10;
division=0:2*pi/n_division:1.8*pi;

points_to_consider=last_point.*ones(n_division,1)+radius*(sin(division)'+cos(division)');
%ypred=predict(gprMdl,points_to_consider);

% set up kernel
sigma=gprMdl.Sigma;
%gamma=mean(std(points_to_consider)); % this comes from matlab documentation
kernelParams=gprMdl.KernelInformation.KernelParameters;
lengthScale=kernelParams(1)
gamma=kernelParams(2)

N=n_division;
kernel = zeros(N);

for i = 1:N
    for j = 1:N
        kernel(i,j) = lengthScale*exp(-gamma*sum((points_to_consider(i,:) - points_to_consider(j,:)).^2)); % Kernel
    end
end

kernel=kernel+ 1e-6*eye(N);

n_kernel=1:size(kernel,1);
F_mi = sfo_fn_mi(kernel,n_kernel);

[A,scores,evals] = sfo_greedy_lazy(F_mi,n_kernel,1);

new_point=points_to_consider(A,:);
xtrain1(n+l,:)=new_point;
new_y=get_y(new_point);
ytrain1(n+l,:)=new_y;


ypred=predict(gprMdl,grid);
surf(X1,X2,reshape(ypred,size(X1)));
title('MI walk prediction');
% hold on
% scatter3(xtrain1(1:n+l,1),xtrain1(1:n+l,2),zeros(n+l,1),k,'filled','k')
% drawnow;
% hold off
pause(0.5)


end

% ypred=predict(gprMdl,grid);
% surf(X1,X2,reshape(ypred,size(X1)));
% title('MI walk prediction');

index2=randsample(length(grid),1);
start=grid(index2,:);
%start=[0 0]

xtrain2=zeros(n+k,2);
prev=start;

for i=2:n+k
   dx=2*pi*rand();
   move=[sin(dx) cos(dx)]; 
   x_new=prev+move;
   prev=x_new;
   xtrain2(i,:)=x_new;
end

ytrain2=get_y(xtrain2);
gprMdl2=fitrgp(xtrain2,ytrain2,'FitMethod','exact','PredictMethod','exact');    

% gprMdl2=fitrgp(xtrain_all_random,ytrain_all_random,'FitMethod','exact','PredictMethod','exact',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,...
%     'MaxObjectiveEvaluations',10));

ypred2=predict(gprMdl2,grid);

figure();
surf(X1,X2,reshape(ypred2,size(X1)));
title('random points');
error_MI=sum((ypred-ytrue).^2)
error_random=sum((ypred2-ytrue).^2)






%%


close all;
clear all;

%get_y= @(x) x(:,1).*exp(-x(:,1).^2-x(:,2).^2);
get_y=@(x) x(:,1)+x(:,2);

% set up the grid
[X1 X2]=meshgrid(-5:0.5:5,-5:0.5:5);
x1_grid=reshape(X1,[],1);
x2_grid=reshape(X2,[],1);
grid=[x1_grid,x2_grid];

% sample some initial training point
radius=1;
n=10;
k=10;
index=randsample(length(grid),n);
xtrain=zeros(n+k,2);
ytrain=zeros(n+k,1);
xtrain(1:n,:)=grid(index,:); 
ytrain(1:n,:)=get_y(xtrain(1:n,:));
kernel=zeros(n+k);

for l=1:k
    
gprMdl=fitrgp(xtrain(1:n+l-1,:),ytrain(1:n+l-1,:),'kernelFunction','squaredexponential');  
% gprMdl=fitrgp(xtrain,ytrain,'FitMethod','exact','PredictMethod','exact',...
%     'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0,...
%     'MaxObjectiveEvaluations',10));
sigma=gprMdl.Sigma;
kernelParams=gprMdl.KernelInformation.KernelParameters;
lengthScale=kernelParams(1);
gamma=kernelParams(2);

% set up possilbe points to take next
last_point=xtrain(n+l-1,:);
n_division=10;
division=0:2*pi/n_division:1.8*pi;
points_to_consider=last_point.*ones(n_division,1)+radius*(sin(division)'+cos(division)');

new_point_cov=zeros(n_division);

for i=1:n_division
    for j = 1:n_division
        new_point_cov(i,j)=lengthScale*exp(-gamma*sum((points_to_consider(i,:)- points_to_consider(j,:)).^2));
    end
end

new_point_cov=new_point_cov+sigma*eye(n_division);

width=1:size(new_point_cov,1);
F_mi = sfo_fn_mi(new_point_cov,width);

[A,scores,evals] = sfo_greedy_lazy(F_mi,width,1);

new_point=points_to_consider(A,:);
xtrain(n+l,:)=new_point;
new_y=get_y(new_point);
ytrain1(n+l,:)=new_y;

ypred=predict(gprMdl,grid);
surf(X1,X2,reshape(ypred,size(X1)));
title('MI walk prediction');
 hold on
% scatter3(xtrain(1:n+l,1),xtrain(1:n+l,2),zeros(n+l,1),k,'filled','k')
 drawnow;
 hold off
pause(0.5)

end

figure();
scatter3(xtrain(1:n+k,1),xtrain(1:n+k,2),zeros(n+k,1),k,'filled','k');





