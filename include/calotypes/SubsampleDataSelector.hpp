#pragma once

#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include <algorithm>

#include "calotypes/DataSelector.hpp"

namespace calotypes
{
	
template< class Data >
class SubsampleDataSelector
: public DataSelector<Data>
{
public:

	typedef std::shared_ptr<SubsampleDataSelector> Ptr;
	typedef std::vector<Data> Dataset;
	
	SubsampleDataSelector( unsigned int ss ) 
	: subsample( ss ) {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset) 
	{
		// Catch case where there is not enough data
		unsigned int subSize = std::min<double>( subsetSize, std::floor( data.size()/subsample ) );
		subset.clear();
		subset.reserve( subSize );
		
		unsigned int subSpan = subSize*subsample;
		unsigned int remSpan = data.size() - subSpan;
		
		boost::random::random_device rng;
		boost::random::uniform_int_distribution<> dist( 0, remSpan );
		unsigned int shift = dist( rng );
		
		unsigned int ind = shift;
		for( unsigned int i = 0; i < subSize; i++ )
		{
			subset.push_back( data[ind] );
			ind += subsample;
		}
		
	}
	
protected:
	
	unsigned int subsample;

};
	
} // end namespace calotypes
