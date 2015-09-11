% Generates plots for results. Assumes matrix format of
% mat(i,j,k) where 
%   i corresponds to trial
%   j corresponds to budget
%   k corresponds to camera

%% Plot parameters
urColor = [1 0 0]; urMarker = 'x';
ssColor = [0 1 0]; ssMarker = 's';
gcColor = [0 0 1]; gcMarker = 'o';
set( 0, 'DefaultTextInterpreter', 'latex' );
budgetLabels = cellfun(@num2str, num2cell(budgets), 'UniformOutput', false);

%% Bar plot of test MSE for all trials
% TODO Average over cameras
figure;
errStdDevs = sqrt( [urVarTestErr, ssVarTestErr, gcVarTestErr] );
[mseBars, mseVarLines] = barwitherr( errStdDevs, [urMeanTestErr, ssMeanTestErr, gcMeanTestErr] );
legend( 'ur', 'ss', 'gc', 'location', 'best' );
set( gca, 'xticklabel', budgetLabels );
xlabel( 'Training Budget $B$' );

set( mseBars(1), 'facecolor', urColor );
set( mseBars(2), 'facecolor', ssColor );
set( mseBars(3), 'facecolor', gcColor );

set( mseBars(1), 'edgecolor', 'none' );
set( mseBars(2), 'edgecolor', 'none' );
set( mseBars(3), 'edgecolor', 'none' );

set( mseVarLines(1), 'linewidth', 1 );
set( mseVarLines(2), 'linewidth', 1 );
set( mseVarLines(3), 'linewidth', 1 );

% TODO Set colors of bars and stuff

%% Scatter plot of CSD versus test error for fixed budget
figure;
axis;
hold on;
budgetInd = 3;
plot( urCSD(budgetInd,:), urTestErrs(budgetInd,:), 'linestyle', 'none', ...
    'marker', urMarker, 'color', urColor );
plot( ssCSD(budgetInd,:), ssTestErrs(budgetInd,:), 'linestyle', 'none', ...
    'marker', ssMarker, 'color', ssColor );
plot( gcCSD(budgetInd,:), gcTestErrs(budgetInd,:), 'linestyle', 'none', ...
    'marker', gcMarker, 'color', gcColor );
xlabel( 'Training Subset CSD' );
ylabel( 'Mean Reprojection Error' );
legend( 'ur', 'ss', 'gc', 'location', 'best' );

%% Plot of CSD/error versus natural training dataset size
% TODO

%% Plot of mean CSD versus training budget size
figure;
csdStdDevs = sqrt( [urVarCSD, ssVarCSD, gcVarCSD] );
[csdBars, csdVarLines] = barwitherr( csdStdDevs, [urMeanCSD, ssMeanCSD, gcMeanCSD] );
legend( 'ur', 'ss', 'gc', 'location', 'best' );
set( gca, 'xticklabel', budgetLabels );
xlabel( 'Training Budget $B$' );
ylabel( 'Training Subset CSD' );

set( csdBars(1), 'facecolor', urColor );
set( csdBars(2), 'facecolor', ssColor );
set( csdBars(3), 'facecolor', gcColor );

set( csdBars(1), 'edgecolor', 'none' );
set( csdBars(2), 'edgecolor', 'none' );
set( csdBars(3), 'edgecolor', 'none' );

set( csdVarLines(1), 'linewidth', 1 );
set( csdVarLines(2), 'linewidth', 1 );
set( csdVarLines(3), 'linewidth', 1 );

%% Produce matrix of parameter variances for each method

budgetInd = 3;
urParams = [ urFx(budgetInd,:);
             urFy(budgetInd,:);
             urPx(budgetInd,:);
             urPy(budgetInd,:);
             urK1(budgetInd,:);
             urK2(budgetInd,:);
             urP1(budgetInd,:);
             urP2(budgetInd,:) ];
         
ssParams = [ ssFx(budgetInd,:);
             ssFy(budgetInd,:);
             ssPx(budgetInd,:);
             ssPy(budgetInd,:);
             ssK1(budgetInd,:);
             ssK2(budgetInd,:);
             ssP1(budgetInd,:);
             ssP2(budgetInd,:) ];
         
gcParams = [ gcFx(budgetInd,:);
             gcFy(budgetInd,:);
             gcPx(budgetInd,:);
             gcPy(budgetInd,:); 
             gcK1(budgetInd,:);
             gcK2(budgetInd,:);
             gcP1(budgetInd,:);
             gcP2(budgetInd,:) ];

urVarParams = var( urParams, 0, 2 );
ssVarParams = var( ssParams, 0, 2 );
gcVarParams = var( gcParams, 0, 2 );

allVars = [ urVarParams';
            ssVarParams';
            gcVarParams' ];

%% Calculate overlap in training data
[numBudgets, numIters] = size( ur );

urAvgOverlap = zeros(numBudgets,1);
ssAvgOverlap = zeros(numBudgets,1);
gcAvgOverlap = zeros(numBudgets,1);

for b = 1:numBudgets
    
    Kur = CalculateDataOverlap( ur(b,:) ) - eye(numIters);
    urAvgOverlap(b) = sum(Kur(:))/( (numIters-1)*numIters );
    
    Kss = CalculateDataOverlap( ss(b,:) ) - eye(numIters);
    ssAvgOverlap(b) = sum(Kss(:))/( (numIters-1)*numIters );
    
    Kgc = CalculateDataOverlap( gc(b,:) ) - eye(numIters);
    gcAvgOverlap(b) = sum(Kgc(:))/( (numIters-1)*numIters );
    
end

figure;
axis;
hold on;
plot( budgets, urAvgOverlap, 'color', urColor, 'marker', urMarker );
plot( budgets, ssAvgOverlap, 'color', ssColor, 'marker', ssMarker );
plot( budgets, gcAvgOverlap, 'color', gcColor, 'marker', gcMarker );
legend( 'ur', 'ss', 'gc', 'location', 'best' );

xlabel( 'Training Budget $B$' );
ylabel( 'Training Subset Overlap' );
