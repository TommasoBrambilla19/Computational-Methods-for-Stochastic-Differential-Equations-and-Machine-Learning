clc;
clear;
close all;

% Parameters
volatilityFactor = 100;
interestRate = 0.04;
maturity = 0.75;
initialY = -1;
initialS = 100;
strike = initialS;
correlation = -0.3;

simulations = 1e3;
steps = 1e4;
deltaT = maturity / steps;

terminalValues = zeros(1, simulations);

for sim = 1:simulations

    dW = sqrt(deltaT) * randn(1, steps);
    dX = sqrt(deltaT) * randn(1, steps);
    dZ = correlation .* dW + sqrt(1 - correlation^2) .* dX;

    asset = initialS;
    vol = initialY;

    for t = 2:steps
        driftTerm = 1 + interestRate * deltaT;
        diffusion = exp(vol) * dW(t);
        asset = asset * (driftTerm + diffusion);

        volDrift = -volatilityFactor * (2 + vol);
        volDiffusion = 0.4 * sqrt(volatilityFactor) * dZ(t);
        vol = vol + (volDrift + 0.4 * sqrt(volatilityFactor) * sqrt(1 - correlation^2)) * deltaT + volDiffusion;
    end

    terminalValues(sim) = asset;
end

payoffs = max(terminalValues - strike, 0);
optionPrice = exp(-interestRate * maturity) * mean(payoffs)
