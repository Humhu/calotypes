#pragma once

#include "calotypes/PointStructure.h"

namespace calotypes
{
	
	class PlanarGridStructure: public PointStructure
	{
	public:
		
		/*! \brief Create a flat grid of size rows by cols with points spaced by dim. */
		PlanarGridStructure( unsigned int rows, unsigned int cols, double dim )
			: gridSize( cols, rows )
		{
			for( unsigned int i = 0; i < rows; i++ )
			{
				for( unsigned int j = 0; j < cols; j++ )
				{
					cv::Point3f point( i*dim, j*dim, 0 );
					points.push_back( point );
				}
			}
		}
		
		cv::Size GetSize() const { return gridSize; }
		
		/*! \brief Returns the points row by row. */
		virtual std::vector< cv::Point3f > GetPoints() const
		{
			return points;
		}
		
	protected:
		
		std::vector< cv::Point3f > points;
		cv::Size gridSize;
		
	};
	
}
