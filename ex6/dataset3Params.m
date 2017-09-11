function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
possible = [0.01 0.03 0.1 0.3 1 3 10 30];
C = 0; 
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
tempC = 0;
tempSigma = 0;

% Define an error value greater than 1 so that it is possible to find a min
error = 100;

% Iterate over the same set twice
for c = 1:size(possible, 2)
	for s = 1:size(possible, 2)
		% Choose our C and sigma to test
		tempC = possible(c);
		tempSigma = possible(s);

		% We need to train the model on X and y
		% However, we only predict with our cross validation set
		[model] = svmTrain(X, y, tempC, @(x1, x2) gaussianKernel(x1, x2, tempSigma));
		[pred] = svmPredict(model, Xval);

		%The error is found from how well the model predicts the CV set
		isError = mean(double(pred ~= yval));

		% Find the minimum error
		if(isError < error)
			C = tempC;
			sigma = tempSigma;
			error = isError
		end
		disp((c - 1)*size(possible, 2) + s);
	end
end
% =========================================================================

end
