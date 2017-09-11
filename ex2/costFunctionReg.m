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


%Cost function
%Find vectorized hypothesis with sigmoid function
hypothesis = sigmoid(X * theta);

%Create vectorized version of cost sum which will be summed in one command
delta = (-1 .* y .* log(hypothesis)) .- ((1 .- y) .* log(1 .- hypothesis));
J = (1/m) * sum(delta);

%Create regularized epsilon without the first theta and add it to the cost
thetaMod = theta;
thetaMod(1) = 0; 
epsilon = (lambda/(2*m)) * sum(thetaMod .^ 2);

J = J + epsilon;

%Gradient with epsilon 
for i = 1:size(X, 2)
	delta = (1/m) * sum((hypothesis .- y) .* X(:, i));
	grad(i) = delta;
	if(i > 1)
		epsilon = (lambda/m) * theta(i);
		grad(i) = grad(i) + epsilon;
	end
end


% =============================================================

end
