#pragma once

#include "calotypes/CameraCalibration.h"

namespace calotypes
{

/*! \brief Interface class for selecting a subset of data. */
template <class Data>
class DataSelector
{
public:
	
	typedef std::vector<Data> Dataset;
	
	DataSelector() {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset) = 0;
};
	
template <class Data>
class DefaultDataSelector
	: public DataSelector<Data>
{
public:
	
	typedef typename DataSelector<Data>::Dataset Dataset;
	
	DefaultDataSelector() {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		subset = data;
	}
};

} // end namespace calotypes
