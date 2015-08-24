#pragma once

#include <boost/random/random_number_generator.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace calotypes
{
	
template<class Engine>
void FisherYatesShuffle( unsigned int numItems, std::vector<unsigned int>& inds, Engine& engine )
{
	inds.resize( numItems );
	for( unsigned int i = 0; i < numItems; i++ )
	{
		inds[i] = i;
	}
	
	unsigned int temp;
	for( unsigned int i = numItems-1; i > 0; i-- )
	{
		boost::random::uniform_int_distribution<> dist( 0, i ); // i inclusive
		unsigned int j = (unsigned int) dist( engine );
		temp = inds[j];
		inds[j] = inds[i];
		inds[i] = temp;
	}
}

} // end namespace calotypes
