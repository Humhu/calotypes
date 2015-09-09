#pragma once

#include "calotypes/DatasetFunctions.hpp"

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
	
/*! \brief Returns the full data. */
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

} // end namespace calotypes
