steps = [1:6];
for i=steps
N = 2^i ;% number of timesteps
randn('state',0);
T = 1;
dt = T/N;
t = 0:dt:T;
a = 0.1; 
b = 0.5;
g=@(x) x;
S0 = 10;
M = 1E6; % number of realisations
S = S0*ones(M,1); % S(0) for all realizations
gS=zeros(M,1);
gST=zeros(M,1);
W = zeros(M,1); % W(0) for all realizations
for j=1:N
dW = sqrt(dt)*randn(M,1); % Wiener increments
S = S + S.*(a*dt+b*dW); % processes at next time step
W = W + dW; % Brownian paths at next step
end
ST = S0*exp( (a-b^2/2)*T + b*W );% exact final value
for k=1:M
    gS(k)=g(S(k));
    gST(k)=g(ST(k));
end
weakError(i) = mean(gS-gST); % weak error
strongError(i) = sqrt(mean((S-ST).^2)); % strong error
end
dt = T./2.^steps;
loglog(dt,abs(weakError),'o--',dt,dt,'*-',dt,abs(strongError),'o-',dt,sqrt(dt),'x-')
legend('Weak Error', 'dt', 'Strong Error', 'sqrt(dt)')
xlabel('dt'); 
ylabel('Error');


