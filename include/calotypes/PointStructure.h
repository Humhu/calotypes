#pragma once

#include <opencv2/core.hpp>

namespace calotypes
{
	
	class PointStructure
	{
	public:
		
		PointStructure() {}
		virtual ~PointStructure() {}
		
		/*! \brief Return an ordered list of the points in this structure. */
		virtual std::vector< cv::Point3f > GetPoints() const = 0;
	};
	
}
