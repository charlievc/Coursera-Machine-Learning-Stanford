function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X*theta;
cost_y1 = (1/m)*(-y.*log(sigmoid(z)));
cost_y2 = (1/m)*((1-y).*log(1-sigmoid(z)));
regterm = (lambda/(2*m))*(theta.^2);
J = sum(cost_y1-cost_y2)+(sum(regterm)-regterm(1));

grad_init = (1/m)*(X'*(sigmoid(z)-y));
grad_temp = grad_init(1);
grad = grad_init + ((lambda/m)*theta);
grad(1) = grad_temp;

% =============================================================

end
