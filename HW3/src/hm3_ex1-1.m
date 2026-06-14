% Monte Carlo simulation for European call option pricing
function [option_price, standard_error] = monte_carlo_option_price(S0, K, r, sigma, T, N)
    rng(42); % For reproducibility
    W_T = randn(N, 1) * sqrt(T);
    S_T = S0 * exp((r - 0.5 * sigma^2) * T + sigma * W_T);
    payoffs = max(S_T - K, 0);
    option_price = exp(-r * T) * mean(payoffs);
    sample_variance = var(payoffs, 1);
    standard_error = sqrt(sample_variance / N);
end

% Finite difference approximation for Delta
function delta = compute_delta(S0, K, r, sigma, T, N, delta_s)
    if nargin < 7
        delta_s = 1e-2;
    end
    f_s = monte_carlo_option_price(S0, K, r, sigma, T, N);
    f_s_plus = monte_carlo_option_price(S0 + delta_s, K, r, sigma, T, N);
    delta = (f_s_plus - f_s) / delta_s;
end

% Stochastic volatility model simulation
function option_price = stochastic_volatility_simulation(alpha, rho, r, T, Y0, S0, K, dt, N)
    rng(42);
    M = round(T / dt);
    Y = zeros(N, M + 1);
    S = zeros(N, M + 1);
    Y(:, 1) = Y0;
    S(:, 1) = S0;
    
    for i = 1:M
        W = randn(N, 1) * sqrt(dt);
        Z = randn(N, 1) * sqrt(dt);
        Z_hat = rho * W + sqrt(1 - rho^2) * Z;
        Y(:, i + 1) = Y(:, i) + (-alpha * (2 + Y(:, i)) + 0.4 * sqrt(alpha) * sqrt(1 - rho^2)) * dt + 0.4 * sqrt(alpha) * Z_hat;
        S(:, i + 1) = S(:, i) .* (1 + r * dt + exp(Y(:, i)) .* W);
    end
    
    payoffs = max(S(:, end) - K, 0);
    option_price = exp(-r * T) * mean(payoffs);
end

% Example usage
S0 = 35; K = 35; r = 0.04; sigma = 0.2; T = 0.5; N = 100000;
[option_price, error] = monte_carlo_option_price(S0, K, r, sigma, T, N);
delta_estimate = compute_delta(S0, K, r, sigma, T, N);

fprintf('Monte Carlo Option Price: %.4f ± %.4f\n', option_price, error);
fprintf('Estimated Delta: %.4f\n', delta_estimate);

% Stochastic Volatility Model Simulation
alpha = 100; rho = -0.3; T = 0.75; Y0 = -1; S0 = 100; K = 100; dt = 0.01;
option_price_sv = stochastic_volatility_simulation(alpha, rho, r, T, Y0, S0, K, dt, N);
fprintf('Option Price under Stochastic Volatility: %.4f\n', option_price_sv);
