clc;
clear;
close all;
tic

% Model parameters
volIntensity = 100;
rate = 0.04;
expiry = 0.75;
initVol = -1;
initPrice = 100;
strikePrice = initPrice;
corrCoeff = -0.3;

% Simulation setups
numPathsList = linspace(1e4, 3e5, 50);
optionEstimates = zeros(size(numPathsList));
varianceSamples = zeros(size(numPathsList));

idx = 1;
for paths = numPathsList
    numPaths = round(paths);
    timeSteps = round(sqrt(numPaths));
    deltaT = expiry / timeSteps;
    timeGrid = 0:deltaT:expiry;

    % Preallocations
    assetVals = [initPrice * ones(numPaths, 1), zeros(numPaths, timeSteps)];
    volVals = [initVol * ones(numPaths, 1), zeros(numPaths, timeSteps)];
    brownian1 = sqrt(deltaT) * randn(numPaths, timeSteps);
    brownian2 = sqrt(deltaT) * randn(numPaths, timeSteps);

    % Time stepping
    for tStep = 1:timeSteps
        prevS = assetVals(:, tStep);
        prevY = volVals(:, tStep);

        correlatedZ = corrCoeff * brownian1(:, tStep) + ...
                      sqrt(1 - corrCoeff^2) * brownian2(:, tStep);

        volDriftConst = 0.4 * sqrt(volIntensity) * sqrt(1 - corrCoeff^2);
        volDiff = 0.4 * sqrt(volIntensity) * correlatedZ;

        assetVals(:, tStep + 1) = prevS + rate * prevS * deltaT + exp(prevY) .* prevS .* brownian1(:, tStep);
        volVals(:, tStep + 1) = prevY + (-volIntensity * (2 + prevY) + volDriftConst) * deltaT + volDiff;
    end

    % Payoff and statistics
    finalPrices = assetVals(:, end);
    payoffs = max(finalPrices - strikePrice, 0);
    optionEstimate = exp(-rate * expiry) * mean(payoffs);
    optionEstimates(idx) = optionEstimate;
    varianceSamples(idx) = var(payoffs);
    idx = idx + 1;
end

toc

% Plotting results
disp(mean(varianceSamples));

figure(1)
plot(numPathsList, optionEstimates, 'b.-')
xlabel('Number of Simulations')
ylabel('Option Price Estimate')
title('Monte Carlo Option Pricing Convergence')
grid on

figure(2)
plot(numPathsList, varianceSamples, 'r.-')
xlabel('Number of Simulations')
ylabel('Sample Variance')
title('Variance vs. Number of Simulations')
grid on
