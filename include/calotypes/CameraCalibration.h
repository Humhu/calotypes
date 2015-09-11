#pragma once

#include <opencv2/core.hpp>
#include <iostream>
#include <Eigen/Geometry>

#include "calotypes/DataSelector.hpp"
#include "calotypes/KernelFunctions.hpp"

#include "calotypes/RandomDataSelector.hpp"

namespace calotypes
{

	// =========================== Basic types =================================
	
	typedef std::vector<cv::Point2f> ImagePoints;
	typedef std::vector<cv::Point3f> ObjectPoints;
	
	struct Pose3D
	{
		typedef Eigen::Matrix<double,6,1> VectorType;
		typedef Eigen::Transform<double, 3, Eigen::Isometry> Transform;
		
		Transform transform;
		
		Pose3D();
		Pose3D( const cv::Mat& transVec, const cv::Mat& rotVec );
		VectorType ToVector() const;
	};
	
	// TODO Implement * and unary operators
	Pose3D operator/( const Pose3D& a, const Pose3D& b );
	
	// TODO const fields and error checking?
	// NOTE The cost of having unused features in the data should be minimal since
	// the containers are dynamically-sized
	struct CameraTrainingData
	{
		std::string name;
		cv::Size imageSize;
		ImagePoints imagePoints;
		ObjectPoints objectPoints;
		
		// 3D Pose
		Pose3D pose;
	};
	
	bool CompareDataName( const CameraTrainingData& a, const CameraTrainingData& b );
	
	// TODO Refactor
	/*! \brief Specifies the camera model complexity. */
	struct CameraTrainingParams
	{
		bool optimizeAspectRatio;
		bool optimizePrincipalPoint;
		bool enableRadialDistortion[3];
		bool enableRationalDistortion[3];
		bool enableTangentialDistortion;
		bool enableThinPrism;
		
		CameraTrainingParams();
	};
	std::ostream& operator<<( std::ostream& os, const CameraTrainingParams& params );
	
	/*! \brief Represents a pinhole camera model. Wraps the various
	 * degrees of correction models that OpenCV supports. */
	class CameraModel
	{
	public:
		
		/*! \brief Initializes with all distortion coefficients. */
		CameraModel();
		
		/*! \brief Sets the distortion coefficient size based on the params. */
		CameraModel( const CameraTrainingParams& params );
		
		CameraTrainingParams trainingParams;
		cv::Mat cameraMatrix;
		cv::Mat distortionCoefficients;
		
	};
	
	std::ostream& operator<<( std::ostream& os, const CameraModel& model );
	
	// ==================== Training and evaluation ============================
	// TODO Add kernel, bandwidth, etc. as parameters
// 	double ImportanceTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
// 									   const cv::Size& imgSize, unsigned int subsetSize,
// 									   DataMetric<CameraTrainingData>::Ptr dataMetric, 
// 									   CameraTrainingParams params = CameraTrainingParams() );
	
	double SubsetTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
								   const cv::Size& imgSize, DataSelector<CameraTrainingData>& selector,
								   unsigned int subsetSize, CameraTrainingParams params = CameraTrainingParams() );
	
	/*! \brief Calibrate a camera model with the specified calibration config. */
	double TrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
								  const cv::Size& imgSize, CameraTrainingParams params = CameraTrainingParams() );

	/*! \brief Test the camera by estimated reprojection error. */
	double TestCameraModel( const CameraModel& model, const std::vector<CameraTrainingData>& data );
	
	// ========================== Data selection ===============================
	typedef DataSelector<CameraTrainingData> CameraDataSelector;
	typedef RandomDataSelector<CameraTrainingData> RandomCameraDataSelector;
	
	void Compute2DFeatures( const ImagePoints& points, Eigen::VectorXd& feats );
	
	// ============================ Metrics ====================================
	
	/*! \brief Full 3D pose distance. */
	class PoseKernelFunction
	: public KernelFunction<CameraTrainingData>
	{
	public:
		
		typedef std::shared_ptr<PoseKernelFunction> Ptr;
		
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
		PoseKernelFunction( const Eigen::Matrix<double,6,6>& w 
						= Eigen::Matrix<double,6,6>::Identity() );
		
		virtual double Difference( const CameraTrainingData& a, const CameraTrainingData& b ) const;
		
	private:
		
		Eigen::Matrix<double,6,6> weights;
	};
	
}
