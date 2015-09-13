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
		std::vector<double> bestDelta;
		std::vector<unsigned int> bestInds;
		while( subset.size() < subsetSize && !available.empty() )
		{
			
// 			std::cout << "Subset is currently: " << std::endl;
// 			for( unsigned int i = 0; i < subset.size(); i++ )
// 			{
// 				std::cout << subset[i].name << " " << subset[i].pose.ToVector().transpose() << std::endl;
// 			}
			
			bestDelta.clear();
			bestInds.clear();
			
			current = metric->delta( subset, available[0] );
			bestDelta.push_back( current );
			bestInds.push_back( 0 );
			
			for( unsigned int i = 1; i < available.size(); i++ )
			{
				current = metric->delta( subset, available[i] );
				
// 				std::cout << "delta for " << available[i].name << " is " << current << std::endl;
				
				if( current > bestDelta[0] ) { continue; }
				// if current == bestDelta[0], we accumulate indices
				if( current < bestDelta[0] )
				{
					bestDelta.clear();
					bestInds.clear();
// 					std::cout << "New best delta for ind " << i << std::endl;
				}
				
				bestDelta.push_back( current );
				bestInds.push_back( i );
			}
			
			// Tie-break randomly
			if( bestDelta.size() > 1 )
			{
				boost::random::uniform_int_distribution<> dist( 0, bestDelta.size() - 1 );
				unsigned int tiebreak = dist( rng );
				bestInds[0] = bestInds[ tiebreak ];
				bestDelta[0] = bestDelta[ tiebreak ];
			}
			
			// Want to minimize - if greedy increases the score, we're done
			if( bestDelta[0] >= 0 ) { return; }
			
			subset.push_back( available[ bestInds[0] ] );
			available.erase( available.begin() + bestInds[0] );
		}
	}
private:
	
	typename DatasetFunction<Data>::Ptr metric;
};

template <class Data>
class RepeatedGreedyDataSelector
: public DataSelector<Data>
{
public:
	
	typedef std::shared_ptr<RepeatedGreedyDataSelector> Ptr;
	typedef typename DataSelector<Data>::Dataset Dataset;

	RepeatedGreedyDataSelector( const typename DatasetFunction<Data>::Ptr& m, unsigned int n = 1 )
	: metric( m ), greedy( m ), numRetries( n ) {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& bestSubset )
	{
		Dataset subset;
		double cost, bestCost = std::numeric_limits<double>::infinity();
		for( unsigned int i = 0; i < numRetries; i++ )
		{
			greedy.SelectData( data, subsetSize, subset );
			cost = (*metric)( subset );
			if( cost < bestCost )
			{
				bestSubset = subset;
				bestCost = cost;
			}
		}	
	}
	
private:
	
	typename DatasetFunction<Data>::Ptr metric;
	GreedyDataSelector<Data> greedy;
	unsigned int numRetries;
	
};

} // end namespace calotypes
