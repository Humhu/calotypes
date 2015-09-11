function [K] = CalculateDataOverlap( tests )

N = numel( tests );
K = zeros( N, N );
for i = 1:N    
    for j = 1:N
        v = coverlap( tests(i).trainingPaths, tests(j).trainingPaths );
        K(i,j) = sum(v)/numel(v);
    end
end