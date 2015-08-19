#include "calotypes/ReprojectionEvaluation.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>

namespace calotypes
{
	
void EvaluateCalibration( const cv::Mat& cameraMatrix, const cv::Mat& distortionCoeffs,
						  const PointStructure& structure,
						  const std::vector< std::vector< cv::Point2f > >& observed,
						  std::vector< std::vector< cv::Point2f > >& predictions )
{
	// First estimate pose with PNP
	std::vector< cv::Point3f > structurePoints = structure.GetPoints();
	predictions = std::vector< std::vector< cv::Point2f > >( observed.size() );
	for( unsigned int i = 0; i < observed.size(); i++ )
	{
		// TODO Use zero distortion?
		cv::Mat rotation, translation;
		bool ret = cv::solvePnP( structurePoints, observed[i], cameraMatrix,
								 distortionCoeffs, rotation, translation );
		// TODO
		if( !ret ) { std::cerr << "Failed PnP!" << std::endl; return; }
	
		// Then reproject to image coordinates
		cv::projectPoints( structurePoints, rotation, translation,
						   cameraMatrix, distortionCoeffs, predictions[i] );
		
		
	}
}

double ComputeMSE( const std::vector< std::vector< cv::Point2f > >& truth,
				   const std::vector< std::vector< cv::Point2f > >& predicted )
{
	double acc = 0;
	unsigned int count = 0;
	
	// TODO
	if( truth.size() != predicted.size() ) { std::cerr << "Sizes do not match." << std::endl; return 0; }
	
	for( unsigned int i = 0; i < truth.size(); i++ )
	{
		const std::vector< cv::Point2f >& truthSet = truth[i];
		const std::vector< cv::Point2f >& predictedSet = predicted[i];
		count += truthSet.size();
		
		if( truthSet.size() != predictedSet.size() ) { std::cerr << "Sizes do not match." << std::endl; return 0; }
		for( unsigned int j = 0; j < truthSet.size(); j++ )
		{
			cv::Point2f diff = truthSet[j] - predictedSet[j];
			acc += std::sqrt( diff.x*diff.x + diff.y*diff.y );
		}
	}
	return acc/count;
}

} // end namespace calotypes
