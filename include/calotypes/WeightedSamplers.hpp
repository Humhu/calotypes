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
		inds.resize( numDraws );
		inds.clear();
		
		double totalWeight = 0;
		for( unsigned int i = 0; i < weights.size(); i++ ) { totalWeight += weights[i]; }
		
		double stride = totalWeight/( numDraws + 1 );
		
		boost::random::uniform_01<> dist;
		double currentPoint = stride*dist( engine );
		
		double accumulatedWeights = weights[0];
		
		unsigned int ind = 0;
		while( inds.size() < numDraws )
		{
			if( accumulatedWeights >= currentPoint )
			{
				inds.push_back( ind );
				currentPoint += stride;
			}
			else
			{
				// Sometimes numerical precision errors cause the last element to not get added
				if( ind == weights.size()-1 ) { 
					inds.push_back( ind );
					break; 
				}
				ind++;
				accumulatedWeights += weights[ind];
			}
		}
	}
};
	
} // end namespace calotypes
