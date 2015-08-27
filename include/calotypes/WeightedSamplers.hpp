#pragma once

#include <boost/random/discrete_distribution.hpp>

namespace calotypes
{

/*! \brief Weighted sampling with replacement. */
template <class Engine>
class NaiveWeightedSampling
{
public:
	
	NaiveWeightedSampling() {}
	
	static void Sample( const std::vector<double>& weights, unsigned int numDraws,
						std::vector<unsigned int>& inds, Engine& engine )
	{
		inds.resize( numDraws );
		boost::random::discrete_distribution<> dist( weights );
		for( unsigned int i = 0; i < numDraws; i++ )
		{
			inds[i] = dist( engine );
		}
	}
};

template <class Engine>
class LowVarianceWeightedSampling
{
public:
	
	LowVarianceWeightedSampling() {}
	
	static void Sample( const std::vector<double>& weights, unsigned int numDraws,
						std::vector<unsigned int>& inds, Engine& engine )
	{
		
	}
};
	
} // end namespace calotypes
