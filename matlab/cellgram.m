function [ res ] = cellgram( func, c1, c2 )

N = numel(c1);
M = numel(c2);

res = cell( N, M );

for i = 1:N
    for j = 1:M
        res{i,j} = func( c1{i}, c2{j} );
    end
end
    