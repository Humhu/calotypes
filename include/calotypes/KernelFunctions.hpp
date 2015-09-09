#pragma once

#include <memory>
#include <Eigen/Dense>
#include <boost/math/distributions/normal.hpp>

namespace calotypes
{
	
/*! \brief Base interface for functions that operate on pairs of data. It must be
 * that kernel(a,b) = kernel(b,a) and kernel(a,a) = 0 */
template <class Data>
class KernelFunction
{
public:

	typedef std::shared_ptr<KernelFunction> Ptr;
	typedef Data DataType;
	
	KernelFunction() {}
	
	/*! \brief Calculates k(a,b) = k(a - b) */
	virtual double Evaluate( const Data& a, const Data& b ) const 
	{
		return Evaluate( Difference( a, b ) );
	}
	
	/*! \brief Calculates a - b */
	virtual double Difference( const Data& a, const Data& b ) const = 0;
	
	/*! \brief Calculates k(x) */
	virtual double Evaluate( double x ) const { return x; }
	
};

/*! \brief Pushes a kernel through a Gaussian PDF. Uses boost's implementation. */
template <class Data>
class GaussianKernelAdaptor
: public KernelFunction<Data>
{
public:
	
	typedef std::shared_ptr<GaussianKernelAdaptor> Ptr;
	
	/*! \brief Construct a Gaussian kernel with specified standard deviation. */
	GaussianKernelAdaptor( typename KernelFunction<Data>::Ptr& k, double s )
	: kernel( k ), normal( 0, s ) {}
	
	virtual double Difference( const Data& a, const Data& b ) const
	{
		return kernel->Difference( a, b );
	}
	
	virtual double Evaluate( double x ) const 
	{
		return boost::math::pdf( normal, x );
	}
	
protected:
	
	
	typename KernelFunction<Data>::Ptr kernel;
	boost::math::normal_distribution<double> normal;
};

/*! \brief Computes the positive-definite kernel (Gram) matrix for a given kernel. 
 * Assumes kernel(a,a) = 0 and kernel(a,b) = kernel(b,a). */
template <class Data>
void ComputeGramMatrix( const KernelFunction<Data>& kernel, const std::vector<Data>& data,
						const Eigen::MatrixXf& K )
{
	unsigned int N = data.size();
	K = Eigen::MatrixXf( N, N );
	
	for( unsigned int i = 0; i < N; i++ ) { K(i,i) = 0; }
	for( unsigned int i = 0; i < N-1; i++ )
	{
		for( unsigned int j = i+1; j < N; j++ )
		{
			K(i,j) = kernel.Evaluate( data[i], data[j] );
			K(j,i) = K(i,j);
		}
	}
}

} // end namespace calotypes
