#pragma once

#include "calotypes/DatasetMetrics.hpp"
#include "calotypes/ProbabilityDensity.hpp"

#include <boost/math/distributions/normal.hpp>
#include <memory>

namespace calotypes
{

/*! \brief Kernel function interface. */
class KernelFunction
{
public:
	
	typedef std::shared_ptr<KernelFunction> Ptr;
	
	KernelFunction() {}
	
	/*! \brief Evaluate the kernel. */
	virtual double operator()( double x ) const = 0;
};

/*! \brief Gaussian PDF kernel using boost's implementation. */
class GaussianKernel
: public KernelFunction
{
public:
	
	typedef std::shared_ptr<GaussianKernel> Ptr;
	
	/*! \brief Construct a Gaussian kernel with specified standard deviation. */
	GaussianKernel( double s )
	: normal( 0, s )
	{}
	
	virtual double operator()( double x ) const
	{
		return boost::math::pdf( normal, x );
	}
	
private:
	
	boost::math::normal_distribution<double> normal;
};

// TODO Allow adding/removing of data
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
	KernelDensityEstimator( const Dataset& d, typename DataMetric<Data>::Ptr m,
							KernelFunction::Ptr k, double h )
	: data( d ), distance( m ), kernel( k )
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
			double x = EvaluateDistance( query, data[i] );
			acc += EvaluateKernel( x * bandwidthReciprocal );
		}
		return acc * bandwidthNormalizer;
	}
	
	/*! \brief This implementation is not normalized. */
	virtual bool IsNormalized() const { return false; }
	
	double inline EvaluateDistance( const Data& a, const Data& b ) const { return (*distance)( a, b ); }
	double inline EvaluateKernel( double x ) const { return (*kernel)( x ); }
	
	
private:
	
	Dataset data;
	typename DataMetric<Data>::Ptr distance;
	KernelFunction::Ptr kernel;
	double bandwidthNormalizer;
	double bandwidthReciprocal;
	
};
	
}
