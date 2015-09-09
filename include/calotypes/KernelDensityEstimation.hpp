#pragma once

#include "calotypes/KernelFunctions.hpp"
#include "calotypes/ProbabilityDensity.hpp"

#include <boost/math/distributions/normal.hpp>
#include <memory>
#include <nanoflann.hpp>

namespace calotypes
{

// TODO Allow adding/removing of data?
/*! \brief Parzen-Rosenblatt type kernel density estimator. */
template <class Data>
class KernelDensityEstimator
: public ProbabilityDensityFunction<Data>
{
public:
	
	typedef std::shared_ptr<KernelDensityEstimator> Ptr;
	typedef std::vector<Data> Dataset;
	
	/*! \brief Construct a KDE with specified data, distance function, kernel function,
	 * and bandwidth. */
	KernelDensityEstimator( const Dataset& d, typename KernelFunction<Data>::Ptr k, double h )
	: data( d ), kernel( k )
	{
		bandwidthNormalizer = 1.0 / ( data.size() * h );
		bandwidthReciprocal = 1.0 / h;
	}
	
	/*! \brief Return an unnormalized PDF estimate. */
	virtual double operator()( const Data& query ) const
	{
		double acc = 0;
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			double x = kernel->Difference( query, data[i] );
			acc += kernel->Evaluate( x * bandwidthReciprocal );
		}
		return acc * bandwidthNormalizer;
	}
	
	/*! \brief This implementation is not normalized. */
	virtual bool IsNormalized() const { return false; }
	
	double inline EvaluateKernel( const Data& a, const Data& b ) const { return (*kernel)( a, b ); }
	
private:
	
	Dataset data;
	typename KernelFunction<Data>::Ptr kernel;
	double bandwidthNormalizer;
	double bandwidthReciprocal;
	
};

}
