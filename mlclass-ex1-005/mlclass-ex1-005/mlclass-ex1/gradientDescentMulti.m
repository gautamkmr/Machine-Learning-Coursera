function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
d = size(X,2); % the number of columns
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    temp_theta = theta;
    for j=1:d
    	sum = 0;
    	for i=1:m
    		x = X(i,:);
    		hypo = transpose(theta)*transpose(x);
    		sum = sum + (hypo - y(i))*x(j);
    	end
    	temp_theta(j) = theta(j) - (alpha*sum)/m;
    end
    
    theta = temp_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %J_history(iter)

end

end
