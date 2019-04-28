function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


%size_theta = size(theta)

%size_X = size(X)


%size_y  = size(y)

%size_lambda = size(lambda)
 
 
%size_grad = size(grad)
 
 
%size_J = size(j)
 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

# loop approach
#sum_0 = 0;  
#for j = 2: length(theta)
#   sum_0 = sum_0 + theta(j)^2;
#end
#regularization_term = lambda/(2*m) * sum_0;
# 
#sum_0 = 0;
#for i = 1 : m
#  z = X(i,:) * theta;
#  hx_i =  1/(1 + exp(-z));
#  sum_0 = sum_0 + [( -y(i) * log(hx_i) - (1 - y(i)) * log(1 - hx_i))];
#end
#J0 = 1/m * sum_0 + regularization_term;

# vectorized approach
regularization_term = lambda/(2*m) * sum(theta(2: end).^2);
z =  X * theta;
hx_i = 1./(1 + exp(-z));
J = 1./m * sum((-y .* log(hx_i)) - (1 - y) .* log(1 - hx_i)) + regularization_term;
% =============================================================

grad = 1/m * sum((hx_i - y) .* X(:, :));
theta1 = theta';
grad(:,2: end) = grad(:, 2: end) + lambda/m .* theta1(:, 2: end);
grad = grad';

end
