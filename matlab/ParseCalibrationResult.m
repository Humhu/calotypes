function [ result ] = ParseCalibrationResult( logPath )
% PARSECALIBRATIONRESULT Parse the output log from a calibration
%   Detailed explanation goes here

fid = fopen( logPath );
if fid == -1
    error( ['Could not open log at ' logPath] );
end

% Get header
result.calibrationTime      = StripLine( fgetl( fid ) );
result.trainingPath         = StripLine( fgetl( fid ) );
result.testPath             = StripLine( fgetl( fid ) );
result.trainingBudget       = str2double( StripLine( fgetl( fid ) ) );
result.datasetSubsample     = str2double( StripLine( fgetl( fid ) ) );
result.method               = StripLine( fgetl( fid ) );
result.kernelStdDev         = str2double( StripLine( fgetl( fid ) ) );

fgetl( fid ); % newline spacing

% Get initial model
fgetl( fid ); % initial model tag
result.initialModel = ParseModel( fid );

fgetl( fid ); % newline spacing

% Get final model
fgetl( fid ); % final model tag
result.finalModel = ParseModel( fid );

fgetl( fid ); % newline spacing

% Get training paths
fgetl( fid ); % training paths tag
result.trainingPaths = {};
while ~feof( fid )
    
    line = fgetl( fid );
    
    if isempty( line )
        break;
    end
    
    result.trainingPaths = { result.trainingPaths{:}, line };
    
end

result.trainingCSD          = str2double( StripLine( fgetl( fid ) ) );
result.trainingError        = str2double( StripLine( fgetl( fid ) ) );
result.testError            = str2double( StripLine( fgetl( fid ) ) );


    function [ stripped ] = StripLine( line )
       inds = strfind( line, ':' );
       stripped = line( (inds(1) + 1):end );
    end

    function [ model ] = ParseModel( fid )
        model.fx = str2double( StripLine( fgetl( fid ) ) );
        model.fy = str2double( StripLine( fgetl( fid ) ) );
        model.px = str2double( StripLine( fgetl( fid ) ) );
        model.py = str2double( StripLine( fgetl( fid ) ) );
        model.aspectRatioOptimized = logical( str2double( StripLine( fgetl( fid ) ) ) );
        model.principalPointOptimized = logical( str2double( StripLine( fgetl( fid ) ) ) );
        model.k1 = str2double( StripLine( fgetl( fid ) ) );
        model.k2 = str2double( StripLine( fgetl( fid ) ) );
        model.k3 = str2double( StripLine( fgetl( fid ) ) );
        model.k4 = str2double( StripLine( fgetl( fid ) ) );
        model.k5 = str2double( StripLine( fgetl( fid ) ) );
        model.k6 = str2double( StripLine( fgetl( fid ) ) );
        model.p1 = str2double( StripLine( fgetl( fid ) ) );
        model.p2 = str2double( StripLine( fgetl( fid ) ) );
        model.s1 = str2double( StripLine( fgetl( fid ) ) );
        model.s2 = str2double( StripLine( fgetl( fid ) ) );
        model.s3 = str2double( StripLine( fgetl( fid ) ) );
        model.s4 = str2double( StripLine( fgetl( fid ) ) );
    end

end

