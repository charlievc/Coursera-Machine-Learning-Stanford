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

h = X*theta;
error = h-y;
mse = error.^2;

cost = (1/(2*m))*sum(mse);
cost_regterm = (lambda/(2*m))*(sum(theta.^2)-theta(1).^2);

% J = (1/2m)*summation((h-y)^2) + (lambda/2m)*summation(theta^2) w/o bias
J = cost+cost_regterm; 


grad = (1/m)*(X'*error);
grad_bias = grad(1);
grad_regterm = (lambda/m)*theta;

% grad = (1/m)*summation((h-y)x) + (lambda/m)*summation(theta) w/o bias
grad = grad + grad_regterm;
grad(1) = grad_bias;

% =========================================================================

grad = grad(:);

end
