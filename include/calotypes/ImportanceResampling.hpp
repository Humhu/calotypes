#pragma once

#include "calotypes/KernelDensityEstimation.hpp"
#include "calotypes/WeightedSamplers.hpp"
#include "calotypes/DataSelector.hpp"

#include <boost/random/random_device.hpp>
#include <boost/random/mersenne_twister.hpp>

namespace calotypes
{

// TODO Possibly expose ability to seed the engine directly, or give an engine reference?
/*! \brief Performs importance resampling to redraw samples from a set drawn according
 * to a proposal distribution to approximate a target distribution. */
template < class Data,
		   class Engine = boost::random::mt19937,
		   class Resampler = NaiveWeightedSampling<Engine> >
void ImportanceResample( const std::vector<Data>& samples,
						 const typename ProbabilityDensityFunction<Data>::Ptr& proposal,
						 const typename ProbabilityDensityFunction<Data>::Ptr& target,
						 unsigned int numSamples, std::vector<Data>& resampled )
{
	// 1. Calculate resampling weights according to target(x)/proposal(x)
	std::vector<double> weights( samples.size() );
	for( unsigned int i = 0; i < samples.size(); i++ )
	{
		weights[i] = (*target)( samples[i] ) / (*proposal)( samples[i] );
	}
	
	// 2. Resample based on the weights
	Engine engine;
	boost::random::random_device rng;
	engine.seed( rng );
	std::vector<unsigned int> resampleIndices;
	Resampler::Sample( weights, numSamples, resampleIndices, engine );
	
	// 3. Return the samples
	// TODO Verify that numSamples = resampleIndices.size()?
	resampled.resize( numSamples );
	for( unsigned int i = 0; i < numSamples; i++ )
	{
		resampled[i] = samples[ resampleIndices[i] ];
	}
}

template <class Data>
class ImportanceDataSelector
: public DataSelector<Data>
{
public:
	
	typedef std::shared_ptr<ImportanceDataSelector> Ptr;
	typedef std::vector<Data> Dataset;
	
	ImportanceDataSelector( const typename ProbabilityDensityFunction<Data>::Ptr& prop,
							const typename ProbabilityDensityFunction<Data>::Ptr& tar )
	: proposal( prop ), target( tar ) {}
	
	virtual void SelectData( const Dataset& data, unsigned int subsetSize, Dataset& subset )
	{
		ImportanceResample( data, proposal, target, subsetSize, subset );
	}
	
private:
	
	typename ProbabilityDensityFunction<Data>::Ptr proposal;
	typename ProbabilityDensityFunction<Data>::Ptr target;
};

} // end namespace calotypes
