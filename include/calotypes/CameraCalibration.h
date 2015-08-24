#pragma once

#include <opencv2/core.hpp>
#include <iostream>
#include "calotypes/DataSelector.hpp"

namespace calotypes
{

	typedef std::vector<cv::Point2f> ImagePoints;
	typedef std::vector<cv::Point3f> ObjectPoints;
	
	// TODO const fields and error checking?
	struct CameraTrainingData
	{
		ImagePoints imagePoints;
		ObjectPoints objectPoints;
	};
	
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
	
	double SubsetTrainCameraModel( CameraModel& model, const std::vector<CameraTrainingData>& data,
								   const cv::Size& imgSize, DataSelector<CameraTrainingData>& selector,
								   unsigned int subsetSize, CameraTrainingParams params = CameraTrainingParams() );
	
	/*! \brief Calibrate a camera model with the specified calibration config. */
	double TrainCameraModel( CameraModel& model, const std::vector<CameraTrainingData>& data,
								  const cv::Size& imgSize, CameraTrainingParams params = CameraTrainingParams() );

	/*! \brief Test the camera by estimated reprojection error. */
	double TestCameraModel( const CameraModel& model, const std::vector<CameraTrainingData>& data );
	
}
