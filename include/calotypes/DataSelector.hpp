#pragma once

#include "calotypes/DatasetMetrics.hpp"

#include <boost/foreach.hpp>
#include <memory>

namespace calotypes
{

// TODO Constify
/*! \brief Interface class for selecting a subset of data. */
template <class Data>
class DataSelector
{
public:
	
	typedef std::shared_ptr<DataSelector> Ptr;
	typedef std::vector<Data> Dataset;
	
	DataSelector() {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset) = 0;
};
	
template <class Data>
class DefaultDataSelector
: public DataSelector<Data>
{
public:
	
	typedef std::shared_ptr<DefaultDataSelector> Ptr;
	typedef typename DataSelector<Data>::Dataset Dataset;
	
	DefaultDataSelector() {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		subset = data;
	}
};

/*! \brief Greedily selects data to subset size. Note that it always returns the
 * max subset size if possible and does not stop when the subset metric decreases. */
template <class Data>
class GreedyDataSelector
: public DataSelector<Data>
{
public:

	typedef std::shared_ptr<GreedyDataSelector> Ptr;
	typedef typename DataSelector<Data>::Dataset Dataset;

	GreedyDataSelector( const typename DatasetMetric<Data>::Ptr& m ) 
	: metric( m ) {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		subset.clear();
		subset.reserve( subsetSize );
		subset.push_back( data[0] ); // Always start with first datapoint
		
		Dataset available( data.begin() + 1, data.end() ); // Remaining data
		
		double bestMetric, currMetric;
		typename Dataset::iterator bestIter, iter;
		while( subset.size() < subsetSize && !available.empty() )
		{
			bestMetric = -std::numeric_limits<double>::infinity();
			for( iter = available.begin(); iter != available.end(); iter++ )
			{
				currMetric = metric->delta( subset, *iter );
				if( currMetric > bestMetric )
				{
					bestMetric = currMetric;
					bestIter = iter;
				}
			}
			subset.push_back( *bestIter );
			available.erase( bestIter );
		}
	}
private:
	
	typename DatasetMetric<Data>::Ptr metric;
};

} // end namespace calotypes
