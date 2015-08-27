#pragma once

#include <opencv2/core.hpp>
#include <iostream>

#include "calotypes/DataSelector.hpp"
#include "calotypes/DatasetMetrics.hpp"

#include "calotypes/RandomDataSelector.hpp"

namespace calotypes
{

	// =========================== Basic types =================================
	
	typedef std::vector<cv::Point2f> ImagePoints;
	typedef std::vector<cv::Point3f> ObjectPoints;
	
	// TODO const fields and error checking?
	// NOTE The cost of having unused features in the data should be minimal since
	// the containers are dynamically-sized
	struct CameraTrainingData
	{
		std::string name;
		cv::Size imageSize;
		ImagePoints imagePoints;
		ObjectPoints objectPoints;
		
		// PNP features
		cv::Mat transVec;
		cv::Mat rotVec;
		
		// 2D features
		Eigen::VectorXd pointFeats;
	};
	
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
	double ImportanceTrainCameraModel( CameraModel& model, std::vector<CameraTrainingData>& data,
									   const cv::Size& imgSize, unsigned int subsetSize,
									   DataMetric<CameraTrainingData>::Ptr dataMetric, 
									   CameraTrainingParams params = CameraTrainingParams() );
	
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
	typedef DataMetric<CameraTrainingData> CameraDataMetric;
	typedef DatasetMetric<CameraTrainingData> CameraDatasetMetric;
	
	/*! \brief Full 3D pose distance. */
	class PoseDistance
	: public CameraDataMetric
	{
	public:
		
		PoseDistance( const CameraModel& m, const cv::Mat& w = cv::Mat::ones( 2, 1, CV_64F ) );
		
		virtual double operator()( const CameraTrainingData& a, const CameraTrainingData& b ) const;
		
	private:
		
		CameraModel model; // Needed for PNP
		cv::Mat weights;
	};
	
	/*! \brief Euclidean distance between feature vectors. */
	class ImageFeatureDistance
	: public CameraDataMetric
	{
	public:
		
		ImageFeatureDistance();
		
		virtual double operator()( const CameraTrainingData& a, const CameraTrainingData& b ) const;
		
	};
	
	typedef AverageDatasetMetric<CameraTrainingData> AverageCameraDatasetMetric;
	typedef MaxDatasetMetric<CameraTrainingData> MaxCameraDatasetMetric;
	
}
