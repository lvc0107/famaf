function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


# hypothesis calculus

K = num_labels;
y_vect = eye(K); 
#size_y_vect = size(y_vect)



#size_theta1 = size(Theta1)
#size_theta2 = size(Theta2)

a1 = [ones(m, 1) X];
#size_a1 = size(a1)

z2 = a1 * Theta1';
#size_z2 = size(z2)

m = size(z2, 1);
a2 = [ones(m, 1) sigmoid(z2)];
#size_a2 = size(a2)

z3 = a2 * Theta2';
#size_z3 = size(z3)

a3 = sigmoid(z3); # = to h_theta(x)
#size_a3 = size(a3)

# Regularization term calculus

rows_theta1 = size(Theta1, 1);
columns_theta1 = size(Theta1, 2);

sum_1 = 0;
for j = 1: rows_theta1
    inner_sum = 0;
    for k = 2: columns_theta1
        inner_sum = inner_sum + Theta1(j, k)^2;
    end
    sum_1 = sum_1 + inner_sum;
end    

rows_theta2 = size(Theta2, 1);
columns_theta2 = size(Theta2, 2);

sum_2 = 0;
for j = 1: rows_theta2
    inner_sum = 0;
    for k = 2: columns_theta2
        inner_sum = inner_sum + Theta2(j, k)^2;
    end
    sum_2 = sum_2 + inner_sum;
end 

regularization_term = lambda/(2*m) * (sum_1 + sum_2);


# Cost function calculus
sum_cost = 0;
for i = 1 : m
    sum_temp = 0;
    for j = 1 : K
        y_k = y_vect(j, y(i));
        hx_k = a3(i, j);
        sum_temp = sum_temp + [( -y_k * log(hx_k) - (1 - y_k) * log(1 - hx_k))];
    end
    sum_cost = sum_cost + sum_temp;
end

J = 1/m * sum_cost + regularization_term;

% =========================================================================

# BACK PROPAGATION
# Gradient 

d3 = zeros(size(a3));
#size_d3 = size(d3)
#d32 = zeros(size(a3));
d2 = zeros(size(a2));
#size_d2 = size(d2)

d2_final = d2(: ,2: end);
#size_d2_final = size(d2_final)

#Theta1_grad_size = size(Theta1_grad)
#Theta2_grad_size = size(Theta2_grad)

for i = 1 : m
    d3(i, :) = a3(i, :) - y_vect(y(i), :);
    d2(i, :) = d3(i, :) * Theta2;
    d2_final(i, :) = d2(i, 2: end) .* sigmoidGradient(z2(i,:));
end


P = size(Theta1, 1);
Q = size(Theta1, 2);

for i = 1: P
    for j = 1: Q
        temp = (d2_final'(i,:) * a1(:, j));
        Theta1_grad(i,j) = Theta1_grad(i,j) + temp;
    end
end


P = size(Theta2, 1);
Q = size(Theta2, 2);

for i = 1: P
    for j = 1: Q
        temp = (d3'(i,:) * a2(:, j));
        Theta2_grad(i,j) = Theta2_grad(i,j) + temp;
    end
end
    
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
grad = 1/m * grad;

end
