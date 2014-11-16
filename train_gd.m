function [w,mean_set,s2,M,errRMSTrain] = train_gd(y,x,M,L)

mean_train=mean(x);
var_train=var(x);

s2=mean(var_train);
[dim1,dim2]=size(x);

I=eye(46)*s2;

w=zeros(1,M);

phi=zeros(1,M);

phi(1)=1;


mean_set=zeros(M-1,dim2);
mean_set(1,:)=mean_train;

for i=2:M-1
  R=mean_train*i/M;
  mean_set(i,:)=R;
end

et=1;
for i=1:dim1
% inputVectorNo=floor((dim1-1).*rand(1) + 1);
inputVector=x(i,:);
for j=2:M
     tmp=inputVector-mean_set(j-1,:);   
     tmp=tmp*(inv(I))*tmp';
     coeff=tmp/2;
     phi(j)=exp(-1*coeff);
end;
e=y(i)-(w*phi');        
w=w+(et*e)*phi;
end;

%
%I2=eye(M);
for i=1:dim1
phi=zeros(1,M); 
phi(1)=1;

for j=2:M
     tmp=x(i,:)-mean_set(j-1,:);   
     tmp=tmp*(inv(I))*tmp';
     coeff=tmp/(2);
     phi(j)=exp(-1*coeff);
     
end
t(i)=w*phi';
end

e=(sum((y'-t).^2))/2;
%f=(w'*w)*L/2;

g=2*(e)/dim1;
g=sqrt(g);
errRMSTrain=g;

end