function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	transtheta = transpose(theta);

	sum1 = 0;
	temp_theta1 = 0;
	for i=1:m
		x = X(i,:);
		hypo = transtheta*transpose(x);     
		sum1 = sum1 + (hypo-y(i))*x(1);
	end

	temp_theta1 = theta(1) - (alpha*sum1)/m;


	sum2 = 0;
	temp_theta2 = 0;
	for i=1:m
		x = X(i,:);
		hypo = transtheta*transpose(x);     
		sum2 = sum2 + (hypo-y(i))*x(2);
	end

	temp_theta2 = theta(2) - (alpha*sum2)/m

        theta(1) = temp_theta1;
        theta(2) = temp_theta2; 

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
