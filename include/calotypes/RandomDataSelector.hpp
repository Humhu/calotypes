#pragma once

#include "calotypes/DataSelector.hpp"
#include "calotypes/SubsetSamplers.hpp"

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace calotypes
{

class RandomDataSelector
: public CameraDataSelector
{
public:
	
	RandomDataSelector()
	{
		boost::random::random_device rng;
		generator.seed( rng );
	}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		std::vector< unsigned int > subsetInds;
		BitmapSampling( data.size(), subsetSize, subsetInds, generator );
		subset.resize( subsetSize );
		for( unsigned int i = 0; i < subsetSize; i++ )
		{
			subset[i] = data[ subsetInds[i] ];
		}
	}
	
private:
	
	boost::random::mt19937 generator; // Seeded by a true random_device
    boost::random::uniform_int_distribution<> index_dist;
	
};
	
} // end namespace calotypes
