clear all;
close all;
% HW4 - SF2525
% Code written by Emanuel Ström

% This is a simple implementation of the Gibbs sampler in Ex. 2.3

%% Sample from the SDE

a = 1.0;
b = 0.1;
N = 100;
t = linspace(0,1,N);
x = X(0, a, b, t);

% Help quantities
dx = x(2:end)-x(1:end-1);
dt = t(2:end)-t(1:end-1);
sum_dx2_dt = sum((dx.^2)./dt);
sum_dx = x(end) - x(1);
sum_dt = t(end) - t(1);

% Plot the solution, trend + bounds
figure(1)
hold on
plot(t,x,'k', 'LineWidth',2)
plot(t, a*t, 'r--', 'LineWidth', 2)
fill([t,t(end:-1:1)], [a*t + 2*b*sqrt(t), a*t(end:-1:1)-2*b*sqrt(t(end:-1:1))], 'r', 'FaceAlpha', 0.2)
legend({"Path", "Trend", "margin"})


%% Part 1: Frequentist approach

% Parameters
N = 100;                  % number of time steps
x0 = 0;                   % initial condition
a_true = 1.0;             % true drift
b_true = 0.1;             % true diffusion
T = 1;                    % final time
t = linspace(0, T, N+1);  % time grid from 0 to 1
dt = diff(t);             % time step sizes

% Simulate the sample path from the SDE
rng(1);                   % for reproducibility
dW = sqrt(dt) .* randn(1, N);  % Brownian increments
x = zeros(1, N+1);
x(1) = x0;
for n = 2:N+1
    x(n) = x(n-1) + a_true * dt(n-1) + b_true * dW(n-1);
end

% MLE Estimation from data
dx = diff(x);                             % observed increments
a_ML = sum(dx) / sum(dt);                 % MLE for drift
b_ML = sqrt(sum((dx - a_ML * dt).^2 ./ dt) / N);  % MLE for diffusion

% Prediction at t_{N+1} = 1.5
t_pred = 1.5;
dt_pred = t_pred - t(end);        % = 0.5
mu_pred = x(end) + a_ML * dt_pred;
var_pred = b_ML^2 * dt_pred;
sigma_pred = sqrt(var_pred);

% Sample from predictive distribution
x_pred_sample = mu_pred + sigma_pred * randn();

figure;
hold on;
grid on;
box on;

% Plot observed path
plot(t, x, 'b.-', 'LineWidth', 1.5, 'MarkerSize', 10, 'DisplayName', '$X_t$ observed');

% Plot predicted point (sample from predictive)
plot(t_pred, x_pred_sample, 'ro', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', '$X_{N+1}$ sample');

% Plot predicted mean
plot(t_pred, mu_pred, 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Predicted mean');

% Confidence interval at prediction time
y_conf = [mu_pred - 2*sigma_pred, mu_pred + 2*sigma_pred];
plot([t_pred t_pred], y_conf, 'k--', 'LineWidth', 1.5, 'DisplayName', '$\pm 2\sigma$ interval');

% Vertical line at prediction time
xline(t_pred, 'r--', 'LineWidth', 1.2, 'DisplayName', 'Prediction time');

% Final formatting
legend('Interpreter', 'latex', 'Location', 'best');
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$X_t$', 'Interpreter', 'latex');
title('Frequentist Prediction at $t = 1.5$', 'Interpreter', 'latex');
set(gca, 'FontSize', 12);

% -----------------------------
% Output prediction info
% -----------------------------
fprintf('Prediction of X_{N+1} (t = 1.5):\n');
fprintf('Mean = %.4f\n', mu_pred);
fprintf('Std  = %.4f\n', sigma_pred);
fprintf('Sampled Value = %.4f\n', x_pred_sample);



%% Part 2: Bayesian approach
% Below is code for generating samples from the posterior distribution

% Pick prior variables
sigma0 = 1;
alpha0 = 1;
beta0 = 1;

% Posterior parameter functions for drift
sigma = @(b) 1 / sqrt(1/sigma0.^2 + sum_dt/b.^2);
mu = @(b) sigma(b).^2 / b.^2 * sum_dx;
A_post = @(b) mu(b) + sigma(b) * randn();

% Posterior parameter functions for diffusion
alpha = @(a) alpha0 + (N-1)/2;
beta = @(a) beta0 + 1/2 * (sum_dx2_dt - 2*a*sum_dx + a.^2*sum_dt); % 1/2 * sum(((dx - a*dt).^2)./dt);
B_post = @(a) sqrt(1./gamrnd(alpha(a), 1/beta(a))); 

% Sample from Posterior
Number_samples=2000; %this is the value to change for point 2.4
ab_samples = gibbs(Number_samples, 100, [0; 1.], {A_post, B_post}); % 0,1 as initial points

% Plot histogram of distribution
figure(2);
hist3(ab_samples', [32, 32],'CdataMode','auto') 
title("Posterior Distribution")
xlabel('a') 
ylabel('b')
view(2)
colorbar

%Part 2.5: mean of gibbs samples
mean_a_sample=sum(ab_samples(1,:))/(Number_samples-99)
mean_b_sample=sum(ab_samples(2,:))/(Number_samples-99)

%% Part 3: Comparison
% Todo: Generate predictions of x(1.5) from the posterior predictive distribution
% Todo: Generate predictions of x(1.5) using the ML-estimates of a_ml, b_ml 

%both written in, respectively: Part 2.6 and Part 2

%% Part 2.6 - Posterior Predictive Distribution via Gibbs Sampling

% Settings
num_outer_samples = 500;    % How many times we re-run Gibbs
num_inner_samples = 1000;   % Gibbs steps per run (after burn-in)
burnin = 100;

% Store posterior predictive samples
x_pred_samples = zeros(1, num_outer_samples);

fprintf("Generating posterior predictive samples...\n");
for i = 1:num_outer_samples
    % Run Gibbs sampling
    ab_sample = gibbs(num_inner_samples + burnin, burnin, [0; 1], {A_post, B_post});

    % Take posterior mean of a and b from Gibbs samples
    a_mean = mean(ab_sample(1, :));
    b_mean = mean(ab_sample(2, :));

    % Compute predictive mean and std for X_{N+1}
    mu_pred = x(end) + 0.5 * a_mean;
    sigma_pred = sqrt(0.5) * b_mean;

    % Sample from predictive distribution
    x_pred_samples(i) = mu_pred + sigma_pred * randn();
end

% Plot histogram of predictive samples
figure;
histogram(x_pred_samples, 40, 'Normalization', 'pdf');
hold on;

% Estimate parameters of the best-fit Gaussian
mu_fit = mean(x_pred_samples);
sigma_fit = std(x_pred_samples);

% Create smooth x range for plotting Gaussian curve
x_range = linspace(min(x_pred_samples), max(x_pred_samples), 500);
gaussian_pdf = normpdf(x_range, mu_fit, sigma_fit);

% Plot the Gaussian curve
plot(x_range, gaussian_pdf, 'r', 'LineWidth', 2);

% Labels and title
xlabel('X_{N+1}');
ylabel('Density');
title('Posterior Predictive Distribution at t = 1.5');
legend('Histogram of samples', 'Best-fit Gaussian');
hold off;



%% Functions!

function x = X(x0, a, b, t)  
    % sample from the SDE
    % given initial point x0, drift a, 
    % diffusion b and times t=[t0,t1,..,tN]

    dt = t(2:end)-t(1:end-1);
    dw = randn(size(dt)) .* sqrt(dt);
    w = [0, cumsum(dw)];
    x = x0 + a*(t-t(1)) + b*w;
end


function x = gibbs(N, Nb, x0, X)
    % N steps of Gibbs sampling.
    % N: integer (number of samples)
    % Nb: Burn-in (ignore the first Nb samples)
    % x0: D x 1 vector (Initial values)
    % X: D cells, each X{d} should be a function,
    % that takes in (D-1) x 1 vector x0,
    % and outputs a sample of the conditional of x(d) given
    % x(1)=x0(1),...,x(d-1)=x0(d-1),x(d+1)=x0(d)...,x(D)=x0(D-1).

    D = numel(x0);
    x = zeros(D, N);
    x(:, 1) = x0;
    for n = 2:N
        for d = 1:D
            x(d, n) = X{d}([x(1:d-1,n); x(d+1:end,n-1)]);          
        end
    end
    x = x(:, Nb:end);
end