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
	
typedef DataSelector<CameraTrainingData> CameraDataSelector;

} // end namespace calotypes
