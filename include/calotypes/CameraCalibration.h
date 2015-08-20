#pragma once

#include <opencv2/core.hpp>

#include <iostream>

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
	
	/*! \brief Calibrate a camera model with the specified calibration config. */
	CameraModel TrainCameraModel( const std::vector<CameraTrainingData>& data,
								  const cv::Size& imgSize,
								  CameraTrainingParams params = CameraTrainingParams() );

	double TestCameraModel( const CameraModel& model, const std::vector<CameraTrainingData>& data );
	
}
