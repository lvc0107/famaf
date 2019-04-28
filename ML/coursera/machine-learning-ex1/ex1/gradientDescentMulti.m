function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n  = size(X, 2);
sum = zeros(n, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % calculus for each theta 
    %for j = 1 : n
    %    res = 0;
    %    for i = 1 : m
    %        % inner calculus for hipothesis = theha(x)
    %        hyp = 0;
    %        for l = 1 : n  
    %            hyp = hyp + (theta(l) * X(i,l));
    %        end   
    %        res = res + (hyp - y(i))* X(i,j); 
    %    end   
    %    theta(j) = theta(1) - alpha* 1/m * res;
    %end
    % ============================================================
    
    
    
   
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    for i = 1 : m
      sum1 = sum1 +  (theta(1) + theta(2)*X(i,2) + theta(3)*X(i,3) -  y(i));
      sum2 = sum2 + ((theta(1) + theta(2)*X(i,2) + theta(3)*X(i,3) -  y(i)) * X(i, 2));
      sum3 = sum3 + ((theta(1) + theta(2)*X(i,2) + theta(3)*X(i,3) -  y(i)) * X(i, 3));
    end

    theta(1) = theta(1) - alpha* 1/m * sum1;
    theta(2) = theta(2) - alpha* 1/m * sum2;
    theta(3) = theta(2) - alpha* 1/m * sum3;


    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
