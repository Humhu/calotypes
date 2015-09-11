%% Parses all calibration results from a folder
rootDir = '~/Documents/abb_experiments/calotypes_mini/results';

i = 1:10;
n = 10:10:50;

ur = [];
ss = [];
gc = [];

for ii = i
    for nn = 1:numel(n)
        
        logname = sprintf( '%s/mtval_camera0_ur_n%i_i%i.txt', rootDir, n(nn), ii );
        if isempty(ur) 
            ur = ParseCalibrationResult( logname );
        else
            ur(nn,ii) = ParseCalibrationResult( logname );
        end
        
        logname = sprintf( '%s/mtval_camera0_ss_n%i_i%i.txt', rootDir, n(nn), ii );
        if isempty(ss) 
            ss = ParseCalibrationResult( logname );
        else
            ss(nn,ii) = ParseCalibrationResult( logname );
        end
        
        logname = sprintf( '%s/mtval_camera0_gc_n%i_i%i.txt', rootDir, n(nn), ii );
        if isempty(gc) 
            gc = ParseCalibrationResult( logname );
        else
            gc(nn,ii) = ParseCalibrationResult( logname );
        end
        
    end
end

%% Calculate results
budgets = 10:10:50;
methods = {'ur', 'ss', 'gc' };

body = ...
    ['@TestErrs = reshape( [@.testError], size(@) );' ...
    '@MeanTestErr = mean( @TestErrs, 2 );' ...
    '@VarTestErr = var( @TestErrs, 0, 2 );' ...
    '' ...
    '@CSD = reshape( [@.trainingCSD], size(@) );' ...
    '@MeanCSD = mean( @CSD, 2 );' ...
    '@VarCSD = var( @CSD, 0, 2 );' ...
    '' ...
    '@Models = reshape( [@.finalModel], size(@) );' ...
    '@Fx = reshape( [@Models.fx], size(@) );' ...
    '@Fy = reshape( [@Models.fy], size(@) );' ...
    '@Px = reshape( [@Models.px], size(@) );' ...
    '@Py = reshape( [@Models.py], size(@) );' ...
    '@K1 = reshape( [@Models.k1], size(@) );' ...
    '@K2 = reshape( [@Models.k2], size(@) );' ...
    '@P1 = reshape( [@Models.p1], size(@) );' ...
    '@P2 = reshape( [@Models.p2], size(@) );'];

for element = methods
   
    m = element{:};

    action = strrep( body, '@', m );
    eval( action );
    
end