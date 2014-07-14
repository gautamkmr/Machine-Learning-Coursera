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

n = size(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
    hypo = sigmoid(transpose(theta)*transpose(X(i,:)));
    J = J +  (-y(i)*log(hypo)-(1-y(i))*log(1-hypo));
    grad = grad + transpose((hypo-y(i))*X(i,:));
end 
J = J/m;
grad = grad/m;

% add regularization term
jt = 0;
for i = 2:n
    jt = jt + theta(i)*theta(i);
    grad(i) = grad(i) + (lambda*theta(i))/m; 
end 

jt = (lambda*jt)/(2*m);
J = J +jt;
% =============================================================

end
