function errRMSTest=test_cfs(w,mean_set,x,y,s2,M,L)

[dim1,dim2]=size(x);
t=zeros(1,dim1);

I=eye(46)*s2;

for i=1:dim1
phi=zeros(1,M); 
phi(1)=1;

for j=2:M
     tmp=x(i,:)-mean_set(j-1,:);   
     tmp=tmp*(inv(I))*tmp';
     coeff=tmp/(2);
     phi(j)=exp(-1*coeff);
     
end
t(i)=w'*phi';
end

e=(sum((y'-t).^2))/2;
f=(w'*w)*L/2;

g=2*(e+f)/dim1;
g=sqrt(g);
errRMSTest=g;
end