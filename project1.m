clc;
clear all;
ubitname='keyurjos';
number=36508572;
lambda_cfs=2;
lambda_gd=1;
load('project1_data.mat');
Y=dataset(:,1);
X=dataset(:,2:47);


%Train and Test Maximum Likelyhood 
[w_cfs,means,s2,M1,errRMSTrainCfs]=train_cfs(Y(1:55698),X(1:55698,:),12,lambda_cfs);
rms_cfs=test_cfs(w_cfs,means,X(55698:69623,:),Y(55698:69623,:),s2,M1,lambda_cfs);

%Train and Test Stochaistic Gradient Descent
[w_gd,mean_set,s2,M2,errRMSTrainGd]=train_gd(Y(1:55698),X(1:55698,:),15,lambda_gd);
rms_gd=test_gd(w_gd,mean_set,X(55698:69623,:),Y(55698:69623,:),s2,M2,lambda_gd);


%Print result
fprintf('My ubit name is %s\n',ubitname);
fprintf('My student number is %d\n',number);
fprintf('the model complexity M_cfs is %d\n', M1);
fprintf('the model complexity M_gd is %d\n', M2);
fprintf('the regularization parameters lambda cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters lambda gd is %4.2f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', rms_gd);


