function [overlaps] = coverlap( a, b )
% COVERLAP Counts overlap between cell arrays 
%   Counts the occurences of strings in a in b

overlaps = zeros( numel(a), 1 );
bvec = b(:);
for i = 1:numel(a)
    overlaps(i) =  sum( strcmp( a{i}, bvec ) );
end