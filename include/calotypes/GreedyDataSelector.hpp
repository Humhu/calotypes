#pragma once

#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "calotypes/DataSelector.hpp"

namespace calotypes
{

/*! \brief Greedily selects data to minimize the dataset function up to the subset size. */
template <class Data>
class GreedyDataSelector
: public DataSelector<Data>
{
public:

	typedef std::shared_ptr<GreedyDataSelector> Ptr;
	typedef typename DataSelector<Data>::Dataset Dataset;

	GreedyDataSelector( const typename DatasetFunction<Data>::Ptr& m ) 
	: metric( m ) {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		subset.clear();
		subset.reserve( subsetSize );
		
		boost::random::random_device rng;
		
		Dataset available( data ); // Remaining data
		
		double current;
		std::vector<double> smallest;
		std::vector<unsigned int> best;
		while( subset.size() < subsetSize && !available.empty() )
		{
			smallest.clear();
			best.clear();
			
			current = metric->delta( subset, available[0] );
			smallest.push_back( current );
			best.push_back( 0 );
			
			for( unsigned int i = 1; i < available.size(); i++ )
			{
				current = metric->delta( subset, available[i] );
				if( current > smallest[0] ) { continue; }
				// if current == smallest[0], we accumulate indices
				if( current < smallest[0] )
				{
					smallest.clear();
					best.clear();
				}
				
				smallest.push_back( current );
				best.push_back( i );
			}
			
			// Tie-break randomly
			if( smallest.size() > 1 )
			{
				boost::random::uniform_int_distribution<> dist( 0, smallest.size() - 1 );
				best[0] = best[ dist( rng ) ];
			}
				
			subset.push_back( available[ best[0] ] );
			available.erase( available.begin() + best[0] );
		}
	}
private:
	
	typename DatasetFunction<Data>::Ptr metric;
};

} // end namespace calotypes
