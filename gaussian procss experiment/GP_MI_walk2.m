


close all;
clear all;
clc;

% set up the grid
border=[-10,10];
[X1, X2]=meshgrid(-10:1:10,-10:1:10);
x1_grid=reshape(X1,[],1);
x2_grid=reshape(X2,[],1);
grid=[x1_grid,x2_grid];

true_alpha = 0.1;
true_gamma = 0.01;
grid_length=length(x1_grid);
covmat=zeros(length(x1_grid));
covmat2=zeros(length(x1_grid));


for i=1:grid_length
    for j = 1:grid_length
        covmat(i,j)=true_alpha*exp(-0.5*true_gamma*sum((grid(i,:)- grid(j,:)).^2));
        covmat2(i,j)=-0.5*sum((grid(i,:)- grid(j,:)).^2);
    end
end
covmat=covmat+1e-6*eye(grid_length);
%%
index=1:grid_length;
F_mi = sfo_fn_mi(covmat,index);
n=5;
[A,scores,evals] = sfo_greedy_lazy(F_mi,index,n);
A
opt.greedy_initial_sset=A;
f = mvnrnd(repmat(0,grid_length,1),covmat,1)';

radius=2;
k=10;
xtrain=zeros(n+k,2);
ytrain=zeros(n+k,1);
xtrain(1:n,:)=grid(A,:); 
ytrain(1:n,:)=get_y(xtrain(1:n,:),grid,f);
kernel=zeros(n+k);

%%
for l=n:n+k-1
    
% using matalb function to find the lengthScale,sigma and signal deviation for current model 

%     gprMdl=fitrgp(xtrain(1:l,:),ytrain(1:l,:),'kernelFunction','squaredexponential');  
    gprMdl=fitrgp(xtrain,ytrain,'FitMethod','exact','PredictMethod','exact',...
    'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',1,...
    'MaxObjectiveEvaluations',10));
    sigma=gprMdl.Sigma;
    kernelParams=gprMdl.KernelInformation.KernelParameters;
    lengthScale=kernelParams(1);
    signal_deviation=kernelParams(2);

% set up possilbe points to take next
    last_point=xtrain(l,:);
    n_division=10;
    division=0:2*pi/n_division:1.8*pi;
    points_to_consider=last_point.*ones(n_division,1)+radius*(sin(division)'+cos(division)');
    points_to_consider=round(points_to_consider,1);

% remove points outside the grid

    points_in_grid=[];
    for j=1:n_division
        if points_to_consider(j,1)>=min(border) && points_to_consider(j,1)<=max(border) &&...
        points_to_consider(j,2)>=min(border) && points_to_consider(j,2)<=max(border)
            points_in_grid=vertcat(points_in_grid,points_to_consider(j,:));
        end
    end
    
% update the kernel and find a new point to add     
    points_to_consider=points_in_grid;
    
    [val,index]=get_y(points_to_consider,grid,f);
    current_covmat=signal_deviation^2*exp(covmat2/lengthScale^2);
    current_covmat=current_covmat+sigma*eye(grid_length);
    
    F_mi = sfo_fn_mi(current_covmat,1:grid_length);

    [A,scores,evals] = sfo_greedy_lazy(F_mi,index,1,opt);
    opt.greedy_initial_sset=A;      %update greedy initial sset so the new point is in set
    index2=A(end);

    new_point=grid(index2,:);
    xtrain(l+1,:)=new_point;
    new_y=get_y(new_point,grid,f);
    ytrain(l+1,:)=new_y;
    
    ypred=predict(gprMdl,grid);
    surf(X1,X2,reshape(ypred,size(X1)));
    title('MI walk prediction');
     hold on
    % scatter3(xtrain(1:n+l,1),xtrain(1:n+l,2),zeros(n+l,1),k,'filled','k')
     drawnow;
     hold off
    pause(0.5);

end

    figure();
    scatter(xtrain(1:n+k,1),xtrain(1:n+k,2),k,'filled','k');
    figure();
    plot(xtrain(:,1),xtrain(:,2))
    figure();
    surf(X1,X2,reshape(f,size(X1)));
    title('Ground truth');


