
clear all;
close all;

x1=[0:0.2:2,4:0.2:6];
x2=[0:0.2:2,4:0.2:6];

[X1,X2]=meshgrid(x1,x2);

y=X1.*exp(-X1.^2-X2.^2)+rand*0.2;
%y=X1+X2+rand*50;

x1=reshape(X1,[],1);
x2=reshape(X2,[],1);

X=[x1,x2];
y=reshape(y,[],1);

gprMdl = fitrgp(X,y);

activeset1=true(242,1);
activeset2=false(242,1);
active=vertcat(activeset1,activeset2);

%gprMdl = fitrgp(X,y,'ActiveSet',active','FitMethod','sr','PredictMethod','fic');
%gprMdl = fitrgp(X,y,'ActiveSetSize',20,'ActiveSetMethod','entropy','FitMethod','sr','PredictMethod','fic');

xtest_grid=[0.0:0.2:2.0,4.0:0.2:6.0];
n=length(xtest_grid);
Xtest1=ones(n)*diag(xtest_grid);
Xtest2=Xtest1';
Xtest1_nx1=reshape(Xtest1,[],1);
Xtest2_nx1=reshape(Xtest2,[],1);

Xtest=[Xtest1_nx1,Xtest2_nx1];

[ypred,ysd] = predict(gprMdl,Xtest);
ysd_point=reshape(ysd,size(Xtest1));
ypred_point=reshape(ypred,size(Xtest1));

surf(Xtest1,Xtest2,ysd_point)
figure();
activeX=gprMdl.ActiveSetVectors;
scatter3(activeX(:,1),activeX(:,2),zeros(length(activeX),1))
% figure();
% ypoint=reshape(y,size(Xtest1));
% surf(Xtest1,Xtest2,ypoint)


%%

clear all
tbl = readtable('abalone.data','Filetype','text','ReadVariableNames',false);tbl.Properties.VariableNames = {'Sex','Length','Diameter','Height','WWeight','SWeight','VWeight','ShWeight','NoShellRings'};
%tbl(1:7,:)

tbl=tbl(1:1000,:);

gprMdl = fitrgp(tbl,'NoShellRings','KernelFunction','ardsquaredexponential',...
      'FitMethod','sr','PredictMethod','fic','Standardize',1,'ActiveSet',1:500);
  
  [ypred,ysd] = resubPredict(gprMdl);
  
  figure();
plot(tbl.NoShellRings,'r.');
hold on
plot(ypred,'b');
xlabel('x');
ylabel('y');
legend({'data','predictions'},'Location','Best');
axis([0 1000 0 30]);
hold off;


