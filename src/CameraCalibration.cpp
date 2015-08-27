#include "calotypes/CameraCalibration.h"
#include "calotypes/KernelDensityEstimation.hpp"
#include "calotypes/ImportanceResampling.hpp"

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
	// TODO These are actually different column/row major, but hack it for now
	   << " px: " << cameraMap(2,0) << " py: " << cameraMap(2,1) << std::endl;
	os << "distortion: " << distortionMap.transpose();
	return os;
}

CameraTrainingParams::CameraTrainingParams()
	: optimizeAspectRatio( false ), optimizePrincipalPoint( false ),
	enableRadialDistortion{ false, false, false },
	enableRationalDistortion{ false, false, false }, 
	enableTangentialDistortion( false ), enableThinPrism( false )
	{}

double ImportanceTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
								   const cv::Size& imgSize, unsigned int subsetSize,
								   CameraDataMetric::Ptr dataMetric, CameraTrainingParams params )
{
	
	std::cout << "Creating sampling objects...";
	GaussianKernel::Ptr kernel = std::make_shared<GaussianKernel>( 1.0 );
	KernelDensityEstimator<CameraTrainingData>::Ptr kde = 
		std::make_shared< KernelDensityEstimator<CameraTrainingData> >( data, dataMetric, kernel, 1.0 );
	UniformDensityFunction<CameraTrainingData>::Ptr target = 
		std::make_shared<UniformDensityFunction< CameraTrainingData> >();
	ImportanceDataSelector<CameraTrainingData>::Ptr dataSelector = 
		std::make_shared<ImportanceDataSelector< CameraTrainingData> >( kde, target );
	std::cout << " completed!" << std::endl;
		
	std::vector<CameraTrainingData> subset;
	dataSelector->SelectData( data, subsetSize, subset );
	double err = TrainCameraModel( model, subset, imgSize, params );
	data = subset;
	return err; 

}
	
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
//  			double err = std::sqrt( diff.x*diff.x + diff.y*diff.y );
// 			if( err > worst ) { worst = err; }
		}
	}
	return acc / count;
// 	return worst; // This is the worst
}
	
PoseDistance::PoseDistance( const CameraModel& m, const cv::Mat& w )
: model( m ), weights( w ) {}

double PoseDistance::operator()( const CameraTrainingData& a, const CameraTrainingData& b ) const
{
	cv::Mat rotationA, rotationB;
	cv::Mat translationA, translationB;
	
	if( a.rotVec.empty() || a.transVec.empty() )
	{
		if( !cv::solvePnP( a.objectPoints, a.imagePoints, model.cameraMatrix,
						   model.distortionCoefficients, rotationA, translationA ) )
		{
			std::cerr << "Warning: PNP failed!" << std::endl;
			return std::numeric_limits<double>::infinity(); 
		}
	}
	else
	{
		rotationA = a.rotVec; 
		translationA = a.transVec;
	}
	
	if( b.rotVec.empty() || b.transVec.empty() )
	{
		if( !cv::solvePnP( b.objectPoints, b.imagePoints, model.cameraMatrix,
						   model.distortionCoefficients, rotationB, translationB ) )
		{
			std::cerr << "Warning: PNP failed!" << std::endl;
			return std::numeric_limits<double>::infinity();
		}
	}
	else
	{	
		rotationB = b.rotVec; 
		translationB = b.transVec;
	}
	
	cv::Mat RA, RB;
	cv::Rodrigues( rotationA, RA );
	cv::Rodrigues( rotationB, RB );
	
	cv::Mat residualR = RA.inv()*RB;
	cv::Mat rotationErr;
	cv::Rodrigues( residualR, rotationErr );
	
	cv::Mat translationErr = translationA - translationB;
	cv::Mat err( 2, 1, CV_64F );
	err.at<double>(0) = cv::norm( translationErr );
	err.at<double>(1) = cv::norm( rotationErr );
	return err.dot( weights );
}
	
void Compute2DFeatures( const ImagePoints& points, Eigen::VectorXd& feats )
{
	unsigned int N = points.size();
	Eigen::Matrix<double, 2, Eigen::Dynamic> p( 2, N );
	for( unsigned int i = 0; i < N; i++ )
	{
		p(0,i) = points[i].x;
		p(1,i) = points[i].y;
	}
	
	Eigen::Vector2d mean = p.rowwise().sum();
	mean = mean / N;
	
	p = p.colwise() - mean;
	Eigen::Matrix2d S = p*p.transpose();
	
	feats = Eigen::VectorXd( 5 );
	feats(0) = mean(0);
	feats(1) = mean(1);
	feats(2) = S(0,0);
	feats(3) = S(1,1);
	feats(4) = S(0,1);
}

ImageFeatureDistance::ImageFeatureDistance() {}

double ImageFeatureDistance::operator()( const CameraTrainingData& a, 
										 const CameraTrainingData& b ) const
{
	Eigen::VectorXd diff = a.pointFeats - b.pointFeats;
	return diff.norm();
}
	
} // end namespace calotypes
