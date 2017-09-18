function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
%x1 = [1 2 1]; x2 = [0 4 -1]; 
C_vec=[0.01 0.03 0.1 0.3 1 3 10 30];
C_vec=C_vec(:);
sig_vec=[0.01 0.03 0.1 0.3 1 3 10 30];
sig_vec=sig_vec(:);
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
min=9999;
for(i=1:size(C_vec))
for(j=1:size(sig_vec))
model= svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sig_vec(j))); 
predictions = svmPredict(model, Xval);
diff_min=mean(double(predictions ~= yval));
if(diff_min<min)
min=diff_min;
C=C_vec(i);
sigma=sig_vec(j);
end
end
end
C
sigma
i
j
% =========================================================================

end
