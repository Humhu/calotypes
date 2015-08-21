#include "calotypes/CameraCalibration.h"
#include <opencv2/calib3d.hpp>

#include <Eigen/Dense>

namespace calotypes
{

CameraModel::CameraModel()
	: cameraMatrix( cv::Mat::eye( 3, 3, CV_64F ) ), 
	distortionCoefficients( cv::Mat::eye( 12, 1, CV_64F ) ) 
{}
	
CameraModel::CameraModel( const CameraTrainingParams& params )
	: trainingParams( params ), cameraMatrix( cv::Mat::eye(3, 3, CV_64F ) )
{
	unsigned int numDistortionParams = 4;
	
	if( params.enableRadialDistortion[2] ) { numDistortionParams = 5; }
	
	if( params.enableRationalDistortion[0] || params.enableRationalDistortion[1]
		|| params.enableRationalDistortion[2] ) { numDistortionParams = 8; }
	
	if( params.enableThinPrism ) { numDistortionParams = 12; }
	
	distortionCoefficients = cv::Mat::zeros( numDistortionParams, 1, CV_64F );
}

std::ostream& operator<<( std::ostream& os, const CameraModel& model )
{
	Eigen::Map< const Eigen::MatrixXd > cameraMap( model.cameraMatrix.ptr<double>(), 3, 3 );
	Eigen::Map< const Eigen::MatrixXd > distortionMap( model.distortionCoefficients.ptr<double>(),
												 model.distortionCoefficients.total(), 1 );
	os << "fx: " << cameraMap(0,0) << " fy: " << cameraMap(1,1) 
	   << " px: " << cameraMap(0,2) << " py: " << cameraMap(1,2) << std::endl;
	os << "distortion: " << distortionMap.transpose();
	return os;
}

CameraTrainingParams::CameraTrainingParams()
	: optimizeAspectRatio( false ), optimizePrincipalPoint( false ),
	enableRadialDistortion{ false, false, false },
	enableRationalDistortion{ false, false, false }, 
	enableTangentialDistortion( false ), enableThinPrism( false )
	{}

// TODO TermCriteria
double TrainCameraModel( CameraModel& model, const std::vector<CameraTrainingData>& data,
						 const cv::Size& imgSize, CameraTrainingParams params )
{
	model = CameraModel( params );
	
	int flags = 0;
	
	if( !params.optimizeAspectRatio ) { flags |= cv::CALIB_FIX_ASPECT_RATIO; }
	if( !params.optimizePrincipalPoint ) { flags |= cv::CALIB_FIX_PRINCIPAL_POINT; }
	
	if( !params.enableRadialDistortion[0] ) { flags |= cv::CALIB_FIX_K1; }
	if( !params.enableRadialDistortion[1] ) { flags |= cv::CALIB_FIX_K2; }
	if( !params.enableRadialDistortion[2] ) { flags |= cv::CALIB_FIX_K3; }
	
	if( params.enableRationalDistortion[0] || params.enableRationalDistortion[1]
		|| params.enableRationalDistortion[2] ) 
	{ 
		flags |= cv::CALIB_RATIONAL_MODEL;
		if( !params.enableRationalDistortion[0] ) { flags |= cv::CALIB_FIX_K4; }
		if( !params.enableRationalDistortion[1] ) { flags |= cv::CALIB_FIX_K5; }
		if( !params.enableRationalDistortion[2] ) { flags |= cv::CALIB_FIX_K6; }
	}
	
	if( !params.enableTangentialDistortion ) { flags |= cv::CALIB_ZERO_TANGENT_DIST; }
	
	if( params.enableThinPrism ) { flags |= cv::CALIB_THIN_PRISM_MODEL; }
	
	std::vector< cv::Mat > rvecs, tvecs;
	
	std::vector< ImagePoints > imagePoints( data.size() );
	std::vector< ObjectPoints > objectPoints( data.size() );
	for( unsigned int i = 0; i < data.size(); i++ )
	{
		imagePoints[i] = data[i].imagePoints;
		objectPoints[i] = data[i].objectPoints;
	}
	cv::calibrateCamera( objectPoints, imagePoints, imgSize, model.cameraMatrix, 
						 model.distortionCoefficients, rvecs, tvecs, flags );
	return TestCameraModel( model, data );
}

double TestCameraModel( const CameraModel& model, const std::vector<CameraTrainingData>& data )
{
	double acc = 0;
	unsigned int count = 0;
	std::vector< cv::Point2f > predictions;
	cv::Mat rotation, translation;
	for( unsigned int i = 0; i < data.size(); i++ )
	{
		// First estimate pose with PnP
		// TODO Use zero distortion?
		bool ret = cv::solvePnP( data[i].objectPoints, data[i].imagePoints, 
								 model.cameraMatrix, model.distortionCoefficients, 
								 rotation, translation );
		// TODO
		if( !ret ) 
		{ 
			std::cerr << "Failed PnP!" << std::endl;
			return std::numeric_limits<double>::infinity();
		}
	
		// Then reproject to image coordinates
		cv::projectPoints( data[i].objectPoints, rotation, translation,
						   model.cameraMatrix, model.distortionCoefficients, predictions );
		count += predictions.size();
		for( unsigned int j = 0; j < predictions.size(); j++ )
		{
			cv::Point2f diff = data[i].imagePoints[j] - predictions[j];
			acc += std::sqrt( diff.x*diff.x + diff.y*diff.y );
		}
	}
	return acc / count;
}
	
} // end namespace calotypes
