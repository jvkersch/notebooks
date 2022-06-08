% Gravitational acceleration
global g;
g = 1;
% Tolerance for constraint solver
global tol;
tol = 1e-1;

% Initial conditions.
% Note: q0/p0 must be column vectors.
% Cosine particle
% x0 = 0;
% v0 = 1;
% q0 = [x0; cos(x0)];
% p0 = [v0; -sin(x0)*v0];

% cos(arctan) particle
% x0 = 0.1;
% q0 = [x0; 1/sqrt(1+x0^2)];
% v0 = 0.1;
% p0 = [v0; -x0*v0/(1+x0^2)^(3/2)];

% cos(x+exp(-x)) particle
x0 = 0.0;
q0 = [x0; cos(x0+exp(-x0))];
v0 = -1.0;
p0 = [v0; -sin(x0+exp(-x0))*(1-exp(-x0))];

% Timestep
h = 0.01
% Maximum integration time
tmax = 7
% Number of integration steps
n = floor(tmax/h)

% Constraint function
function res = phi(q)
    x = q(1, :);
    y = q(2, :);
    % res = y - cos(x);
    % res = y - 1./sqrt(1 + x.^2);
    res = y - cos(x + exp(-x));
end

% Gradient of the constraint
function grad = grad_phi(q)
    n = size(q, 2);
    x = q(1, :);
    y = q(2, :);
    % grad = [sin(x); ones(1, n)];
    % grad = [x./(1.+x.^2).^(3/2); ones(1, n)];
    grad = [sin(x+exp(-x)).*(1.-exp(-x)); ones(1, n)];
end

% Potential function
function res = v(q)
    global g
    res = g*q(2, :);
end

% Gradient of the potential
function grad = grad_v(q)
    global g;
    n = size(q, 2);
    grad = repmat([0; g], 1, n);
end

%%%% RATTLE implementation (generic)
%%%% Note: nothing below this line should have to be modified.
function [q2,p1] = update_position(ell, h, q0, p0)
    p1 = p0 - h/2*(grad_v(q0) + ell*grad_phi(q0));
    q2 = q0 + h*p1;
end

function p2 = update_momentum(mu, h, q2, p1)
    p2 = p1 - h/2*(grad_v(q2) + mu*grad_phi(q2));
end

function res = position_constraint(ell, h, q0, p0)
    [q2,p1] = update_position(ell, h, q0, p0);
    res = phi(q2);
end

function res = update_position_multiplier(ell, h, q0, p0)
    global tol
    wrapper = @(x) position_constraint(x, h, q0, p0);
    res = secant(wrapper, ell, ell+1, tol)(:, end);
end

function res = momentum_constraint(mu, h, q2, p1)
    p2 = update_momentum(mu, h, q2, p1);
    res = dot(grad_phi(q2), p2);
end

function res = update_momentum_multiplier(mu, h, q2, p1)
    global tol
    wrapper = @(x) momentum_constraint(x, h, q2, p1);
    res = secant(wrapper, mu, mu+1, tol)(:, end);
end

function [ts,qs,ps] = rattle(h, n, q, p)
    % Initialize multipliers
    ell = 0;
    mu = 0;

    % Output arrays
    qs = zeros(2, n);
    ps = zeros(2, n);
    ts = zeros(1, n);
    
    for k = 1:n
        % One step
        ell = update_position_multiplier(ell, h, q, p);
        [q,p_half] = update_position(ell, h, q, p);
        mu = update_momentum_multiplier(mu, h, q, p_half);
        p = update_momentum(mu, h, q, p_half);

        % Update output arrays
        qs(:, k) = q;
        ps(:, k) = p;
        ts(:, k) = k*h;
    end;
end

function h = hamiltonian(qs, ps)
    h = sum(ps.^2, 1)/2 + v(qs);
end

function res = f1(qs, ps)
    res = phi(qs);
end

function res = f2(qs, ps)
    res = sum(grad_phi(qs).*ps, 1);
end

[ts,qs,ps] = rattle(h, n, q0, p0);

h0 = dot(p0, p0)/2 + v(q0);
delta_h = abs(hamiltonian(qs, ps) - h0);

figure(1)
subplot(221);
plot(qs(1, :), qs(2, :));
grid on;
title('x vs. y');
xlabel('x');
ylabel('y');

subplot(222);
semilogy(ts, delta_h);
title('\Delta H');
grid on;

subplot(223);
plot(ts, abs(f1(qs, ps) - f1(q0, p0)));
title('\Delta f_1');
grid on;

subplot(224);
plot(ts, abs(f2(qs, ps) - f2(q0, p0)));
title('\Delta f_2');
grid on;
