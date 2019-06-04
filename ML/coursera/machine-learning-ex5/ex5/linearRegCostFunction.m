function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

 
function h = hyp(theta, X, i)
   h = X(i, :) * theta;
end
 
% Iterative form 
sum0 = 0;
for i = 1 : m
    sum0 = sum0 + (hyp(theta, X, i) -  y(i))^2;
end

n = length(theta); % number of attributes
sum1 = 0;
for i = 2 : n
    sum1 = sum1 + theta(i)^2;
end
reg_term = lambda/(2*m) * sum1;
J = 1/(2*m) * sum0 + reg_term;


%Vectorized form
reg_term = lambda/(2*m) * sumsq(theta(2:end));
J = 1/(2*m) * (X* theta - y)' * (X* theta - y) + reg_term;
 
% GRADIENT
for j = 1: length(theta)
  sum = 0;
  for i = 1 : m
     sum = sum + (hyp(theta, X, i) - y(i))* X(i, j)';
  end
 
  if j == 1
     grad(j) = (1/m * sum );
  else
     grad(j) = (1/m * sum ) + lambda/m * theta(j);
  end

end

% =========================================================================

end
