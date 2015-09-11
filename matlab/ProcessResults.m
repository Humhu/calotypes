%% Parses all calibration results from a folder
% Results are organized into structs. Each struct has fields:
%   camera_name
%   budget
%   result fields (parsed results)
%
% The structs will be named [ur/ss/gc][Arm/Manual][Validation/Test]
%   ie. urArmValidation
%
% The arm validation struct will be arrayed as
%   a(i,j,k)
%     i - camera index
%     j - budget index
%     k - trial index
%
% The manual validation struct will be arrayed as
%   a(i,j,h,k)
%     i - camera index
%     j - budget index
%     h - set index
%     k - trial index
%
% Files to be processed are named
% [ur/ss/gc]_[validation/test]_[arm/manual_[1-10]]_n[budget]_i[trial].log

rootDir = '~/Documents/abb_experiments/calotypes/partial_validation_b0_5';

trials = 1:10;
manual_sets = 1:10;
budgets = 10:10:50;

methods = {'ur', 'ss', 'gc'};

urArmValidation = [];
urManualValidation = [];
ssArmValidation = [];
ssManualValidation = [];
gcArmValidation = [];
gcManualValidation = [];

dirs = dir( rootDir );
dirs = dirs( [dirs.isdir] );
valid = true( size(dirs) );
for i = 1:numel( dirs )
    if ~isempty( strfind( dirs(i).name, '.' ) )
        valid(i) = 0;
    end
end
dirs = dirs(valid);

% Loop over methods
for mi = 1:numel( methods )
    method = methods{mi};

    eval( [ method, 'ArmValidation = [];' ] );
    eval( [ method, 'ManualValidation = [];' ] );
    
    armValidationResults = [];
    manualValidationResults = [];
    
    % Loop over directories (cameras)
    for i = 1:numel( dirs )
        d = dirs(i);
        midDir = sprintf( '%s/%s', rootDir, d.name );
        
        % Loop over budgets
        for j = 1:numel(budgets)
            b = budgets(j);
            
            % Loop over trials
            for k = trials
                
                % Arm validation
                logname = sprintf( '%s/%s_validation_arm_n%i_i%i.log', ...
                    midDir, method, b, k );
                fprintf( 'Parsing %s...\n', logname );
                parsed = ParseCalibrationResult( logname );
                parsed.camera_name = d.name;
                parsed.budget = b;
                
                if isempty( armValidationResults )
                    armValidationResults = parsed;
                else
                    armValidationResults(i,j,k) = parsed;
                end
                
                % Manual validations
                for h = manual_sets
                    % Arm validation
                    logname = sprintf( '%s/%s_validation_manual_%i_n%i_i%i.log', ...
                        midDir, method, h, b, k );
                    fprintf( 'Parsing %s...\n', logname );
                    parsed = ParseCalibrationResult( logname );
                    parsed.camera_name = d.name;
                    parsed.budget = b;
                    if isempty( manualValidationResults )
                        manualValidationResults = parsed;
                    else
                        manualValidationResults(i,j,h,k) = parsed;
                    end
                end % end set loop
            end % end trial loop
        end % end budget loop
    end % end directory loop
    
    action = '@ArmValidation = armValidationResults;';
    eval( strrep( action, '@', method ) );
    action = '@ManualValidation = manualValidationResults;';
    eval( strrep( action, '@', method ) );
    
end

clear armValidationResults manualValidationResults;

%% Calculate results
budgets = 10:10:50;
methods = {'ur', 'ss', 'gc' };

body = ...
    ['@ArmErrs = reshape( [@ArmValidation.testError], size(@ArmValidation) );' ...
    '@ArmCSDs = reshape( [@ArmValidation.trainingCSD], size(@ArmValidation) );' ...
    '@ArmModels = reshape( [@ArmValidation.finalModel], size(@ArmValidation) );' ...
    ...
    '@ManualErrs = reshape( [@ManualValidation.testError], size(@ManualValidation) );' ...
    '@ManualCSDs = reshape( [@ManualValidation.trainingCSD], size(@ManualValidation) );' ...
    '@ManualModels = reshape( [@ManualValidation.finalModel], size(@ManualValidation) );' ...
    ...
    '@ArmErrMeans = mean( @ArmErrs, 3 );' ...
    ...
    '@ManualErrMeans = mean( mean( @ManualErrs, 4 ), 3 );' ...
    '@ManualErrVars = mean( squeeze( var( @ManualErrs, 0, 3 ) ), 3 );' ...
    ...
    '@ArmCSDMeans = mean( @ArmCSDs, 3 );' ...
    ...
    '@ManualCSDMeans = mean( mean( @ManualCSDs, 4 ), 3 );' ...
    '@ManualCSDVars = mean( squeeze( var( @ManualCSDs, 0, 3 ) ), 3 );' ...
    ...
    '@ManualFx = reshape( [@ManualModels.fx], size(@ManualValidation) );' ...
    '@ManualFy = reshape( [@ManualModels.fy], size(@ManualValidation) );' ...
    '@ManualPx = reshape( [@ManualModels.px], size(@ManualValidation) );' ...
    '@ManualPy = reshape( [@ManualModels.py], size(@ManualValidation) );' ...
     ...
    '@ManualFxVars = mean( squeeze( var( @ManualFx, 0, 3 ) ), 3 );' ...
    '@ManualFyVars = mean( squeeze( var( @ManualFy, 0, 3 ) ), 3 );' ...
    '@ManualPxVars = mean( squeeze( var( @ManualPx, 0, 3 ) ), 3 );' ...
    '@ManualPyVars = mean( squeeze( var( @ManualPy, 0, 3 ) ), 3 );' ]; 

for element = methods
    
    m = element{:};
    
    action = strrep( body, '@', m );
    eval( action );
    
end
