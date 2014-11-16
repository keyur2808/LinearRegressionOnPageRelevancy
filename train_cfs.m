function [w,mean_set,s2,M,errRMSTrain] = train_cfs(y,x,M,L)


mean_train=mean(x);
var_train=var(x);

s2=mean(var_train);
[dim1,dim2]=size(x);

I=eye(46)*s2;

w=zeros(1,M);
phi=zeros(dim1,M);

for k=1:dim1
    phi(k,1)=1;
end

mean_set=zeros(M-1,dim2);
mean_set(1,:)=mean_train;

for i=2:M-1
  R=mean_train*i/M;
  mean_set(i,:)=R;
end

for j=2:M
    for i=1:dim1
     tmp=x(i,:)-mean_set(j-1,:);   
     tmp=tmp*(inv(I))*tmp';
     coeff=tmp/2;
     phi(i,j)=exp(-1*coeff);
    end
end

%phi(:,1)=ones(dim1,1);

I2=eye(M);
w=pinv((L*I2+phi'*phi))*phi'*y;


for i=1:dim1
phiBs=zeros(1,M); 
phiBs(1)=1;

for j=2:M
     tmp=x(i,:)-mean_set(j-1,:);   
     tmp=tmp*(inv(I))*tmp';
     coeff=tmp/(2);
     phiBs(j)=exp(-1*coeff);
     
end
t(i)=w'*phiBs';
end

e=(sum((y'-t).^2))/2;
f=(w'*w)*L/2;

g=2*(e+f)/dim1;
g=sqrt(g);
errRMSTrain=g;


end