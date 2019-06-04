function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta); % number of attributes
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum = 0;
for i = 1 : m
    h_i = sigmoid(theta' * X(i, :)');
    sum = sum + ((y(i) * log(h_i)) + ((1 - y(i))* log(1 - h_i)));
end

sum_theta2 = 0;
for i = 2 : n
    sum_theta2 = sum_theta2 + theta(i)^2;
end

J = - (1/m * sum ) + lambda/(2*m) * sum_theta2;


% GRADIENT
for j = 1: length(theta)
  sum = 0;
  for i = 1 : m
     h_i = sigmoid(theta' * X(i, :)');
     sum = sum + (h_i - y(i))* X(i, j)';
  end
 
  if j == 1
     grad(j) = (1/m * sum );
  else
     grad(j) = (1/m * sum ) + lambda/m * theta(j);
  end

end




% =============================================================

end
