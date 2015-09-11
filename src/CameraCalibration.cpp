#include "calotypes/CameraCalibration.h"
#include "calotypes/KernelDensityEstimation.hpp"
#include "calotypes/ImportanceResampling.hpp"

#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>

namespace calotypes
{

bool CompareDataName( const CameraTrainingData& a, const CameraTrainingData& b )
{
	if( a.name.size() == b.name.size() ) 
	{
		return a.name < b.name;
	}
	return a.name.size() < b.name.size();
}
	
Pose3D::Pose3D()
	: transform( Eigen::Matrix<double,4,4>::Identity() )
{}

Pose3D::Pose3D( const cv::Mat& transVec, const cv::Mat& rotVec )
{
	Eigen::Translation<double,3> trans( transVec.at<double>(0), transVec.at<double>(1), transVec.at<double>(2) );
	Eigen::Vector3d aaBase( rotVec.at<double>(0), rotVec.at<double>(1), rotVec.at<double>(2) );
	double angle = aaBase.norm();
	aaBase.normalize();
	Eigen::AngleAxisd aa( angle, aaBase );
	transform = trans*aa;
}

Pose3D::VectorType Pose3D::ToVector() const
{
	VectorType vec;
	vec(0) = transform.translation().x();
	vec(1) = transform.translation().y();
	vec(2) = transform.translation().z();
	Eigen::Quaterniond q( transform.rotation() );
	Eigen::Vector3d aa = q.vec();
	double angle = std::acos( q.w() )*2;
	if( angle > M_PI ) { angle = angle - 2*M_PI; }
	if( angle < -M_PI ) { angle = angle + 2*M_PI; }
	aa *= angle;
	vec.block<3,1>(3,0) = aa;
	return vec;
}

Pose3D operator/( const Pose3D& a, const Pose3D& b )
{
	Pose3D other;
	other.transform = a.transform * b.transform.inverse();
	return other;
}
	
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
	os << "fx: " << cameraMap(0,0) << std::endl;
	os << "fy: " << cameraMap(1,1) << std::endl;
	os << "px: " << cameraMap(2,0) << std::endl;
	os << "py: " << cameraMap(2,1) << std::endl;
	os << "aspect ratio optimized: " << model.trainingParams.optimizeAspectRatio << std::endl;
	os << "optimize principal point: " << model.trainingParams.optimizePrincipalPoint << std::endl;
	os << "radial distortion k1: " << distortionMap(0) << std::endl;
	os << "radial distortion k2: " << distortionMap(1) << std::endl;
	os << "radial distortion k3: " << (model.trainingParams.enableRadialDistortion[2] ? distortionMap(4) : 0.0) << std::endl;
	os << "radial distortion k4: " << (model.trainingParams.enableRationalDistortion[0] ? distortionMap(5) : 0.0) << std::endl;
	os << "radial distortion k5: " << (model.trainingParams.enableRationalDistortion[1] ? distortionMap(6) : 0.0) << std::endl;
	os << "radial distortion k6: " << (model.trainingParams.enableRationalDistortion[2] ? distortionMap(7) : 0.0) << std::endl;
	os << "tangential distortion p1: " << distortionMap(2) << std::endl;
	os << "tangential distortion p2: " << distortionMap(3) << std::endl;
	os << "thin lens s1: " << (model.trainingParams.enableThinPrism ? distortionMap(8) : 0.0) << std::endl;
	os << "thin lens s2: " << (model.trainingParams.enableThinPrism ? distortionMap(9) : 0.0) << std::endl;
	os << "thin lens s3: " << (model.trainingParams.enableThinPrism ? distortionMap(10) : 0.0) << std::endl;
	os << "thin lens s4: " << (model.trainingParams.enableThinPrism ? distortionMap(11) : 0.0) << std::endl;	
	return os;
}

CameraTrainingParams::CameraTrainingParams()
	: optimizeAspectRatio( false ), optimizePrincipalPoint( false ),
	enableRadialDistortion{ false, false, false },
	enableRationalDistortion{ false, false, false }, 
	enableTangentialDistortion( false ), enableThinPrism( false )
	{}

std::ostream& operator<<( std::ostream& os, const CameraTrainingParams& params )
{
	os << "aspect ratio: " << params.optimizeAspectRatio << std::endl;
	os << "principal point: " << params.optimizePrincipalPoint << std::endl;
	os << "radial distortion k1: " << params.enableRadialDistortion[0] << std::endl;
	os << "radial distortion k2: " << params.enableRadialDistortion[1] << std::endl;
	os << "radial distortion k3: " << params.enableRadialDistortion[2] << std::endl;
	os << "radial distortion k4: " << params.enableRationalDistortion[0] << std::endl;
	os << "radial distortion k5: " << params.enableRationalDistortion[1] << std::endl;
	os << "radial distortion k6: " << params.enableRationalDistortion[2] << std::endl;
	os << "tangential distortion: " << params.enableTangentialDistortion << std::endl;
	os << "thin prism: " << params.enableThinPrism << std::endl;
	return os;
}

	
// double ImportanceTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
// 								   const cv::Size& imgSize, unsigned int subsetSize,
// 								   CameraDataMetric::Ptr dataMetric, CameraTrainingParams params )
// {
// 	
// 	GaussianKernel::Ptr kernel = std::make_shared<GaussianKernel>( 1.0 );
// 	KernelDensityEstimator<CameraTrainingData>::Ptr kde = 
// 		std::make_shared< KernelDensityEstimator<CameraTrainingData> >( data, dataMetric, kernel, 0.1 );
// 	UniformDensityFunction<CameraTrainingData>::Ptr target = 
// 		std::make_shared<UniformDensityFunction< CameraTrainingData> >();
// 	ImportanceDataSelector<CameraTrainingData>::Ptr dataSelector = 
// 		std::make_shared<ImportanceDataSelector< CameraTrainingData> >( kde, target );
// 		
// 	std::vector<CameraTrainingData> subset;
// 	dataSelector->SelectData( data, subsetSize, subset );
// 	double err = TrainCameraModel( model, subset, imgSize, params );
// 	data = subset;
// 	return err; 
// }
	
double SubsetTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
							   const cv::Size& imgSize, DataSelector<CameraTrainingData>& selector,
							   unsigned int subsetSize, CameraTrainingParams params )
{
	std::vector<CameraTrainingData> subset;
	selector.SelectData( data, subsetSize, subset );
	double err = TrainCameraModel( model, subset, imgSize, params );
	data = subset;
	return err;
}
	
// TODO TermCriteria
double TrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
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
	
// 	double worst = -std::numeric_limits<double>::infinity();
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
	
PoseKernelFunction::PoseKernelFunction( const Eigen::Matrix<double,6,6>& w )
: weights( w ) {}

double PoseKernelFunction::Difference( const CameraTrainingData& a, const CameraTrainingData& b ) const
{
	Pose3D diff = a.pose / b.pose;
	Pose3D::VectorType diffVec = diff.ToVector();
	return diffVec.transpose() * weights * diffVec;
}

} // end namespace calotypes
