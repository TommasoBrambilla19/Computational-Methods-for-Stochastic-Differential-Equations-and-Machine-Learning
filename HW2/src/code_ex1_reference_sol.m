
clc
clear
close all
x0 = 1;
Kmax = 10;
N = 2^Kmax;
T = 1;
dt = T/N;
real = 100000;
mu = 0.1;
sigma = 0.5;
a = @(t , x) mu*x;
b = @(t , x) sigma*x;
g = @(x) exp(x);
%g = @(x) x;

X = zeros(1,N+1);
X(1) = x0;
XT = zeros(1, real);
m = 6;
XT2 = zeros(m, real);

W = sqrt(dt)*cumsum(randn(real,N),2);
W = [zeros(real,1) W];

% reference solution
for i = 1:real

    WW = W(i,:);
    t = 0:dt:T;
    for j = 2:N+1
        X(j) = X(j-1) + a(t(j-1) , X(j-1)).*dt + b(t(j-1) , X(j-1)).*(WW(j) - WW(j-1));
    end
    XT(i) = X(1,end);
end

%approximated solution
steps = 1:m;
dts = dt.*2.^steps;
for k = steps
    B = 2^k;
    dt_new = dt*B;

    t = 0:dt_new:T;
    N = length(t)-1;
    X = zeros(1,N+1);
    X(1,1) = x0;
    for i = 1:real
        WW = W(i,1:B:end);
        for j = 2:N+1
            X(j) = X(j-1) + a(t(j-1) , X(j-1)).*dt_new + b(t(j-1) , X(j-1)).*(WW(j) - WW(j-1));
        end
        XT2(k , i) = X(1,end);
    end
end

% errors
strong = zeros(1,steps(end));
weak = zeros(1,steps(end));
for i = steps
    strong(i) = sqrt((mean(XT - XT2(i , :))).^2);
    weak(i) = abs(mean(g(XT)) - mean(g(XT2(i , :))));
end

% plots

% in the plots we want to analyse the convergence
figure(1)
loglog(dts , dts.^(0.5),'*-',dts , strong,'o-')
legend('sqrt(dt)','strong err')
title("strong error")
xlabel('dt'); 
ylabel('Error');

figure(2)
loglog(dts , dts,'*--',dts , weak,'o--');
legend('dt','weak err')
title("weak error")
xlabel('dt'); 
ylabel('Error');


%loglog(dts,weak,'o--',dts,dts,'*-',dts,strong,'o-',dts,sqrt(dts),'x-')
%legend('Weak Error', 'dt', 'Strong Error', 'sqrt(dt)')
%xlabel('dt'); 
%ylabel('Error');






