#pragma once

#include <opencv2/core.hpp>

#include "calotypes/PointStructure.h"

namespace calotypes
{

/*! \brief Evaluate a camera calibration on detected points of a known structure. */
void EvaluateCalibration( const cv::Mat& cameraMatrix, const cv::Mat& distortionCoeffs,
						  const PointStructure& structure,
						  const std::vector< std::vector< cv::Point2f > >& observed,
						  std::vector< std::vector< cv::Point2f > >& predicted );

double ComputeMSE( const std::vector< std::vector< cv::Point2f > >& truth,
				   const std::vector< std::vector< cv::Point2f > >& predicted );

}
