#pragma once

#include "calotypes/KernelFunctions.hpp"

namespace calotypes
{

/*! \brief Base interface for functions that operate on sets of data and return scalars. */
template <class Data>
class DatasetFunction
{
public:
	
	typedef std::shared_ptr<DatasetFunction> Ptr;
	
	typedef Data DataType;
	typedef std::vector<Data> Dataset;
	
	DatasetFunction() {}
	
	/*! \brief Uses a precomputed kernel matrix. */
	virtual double operator()( const Eigen::MatrixXf& K ) const = 0;
	
	/*! \brief Calculates pairwise metrics in place. */
	virtual double operator()( const Dataset& data ) const = 0;
	
	/*! \brief Returns the change in metric associated with adding the specified data. May be faster
	 * than calculating difference directly. */
	virtual double delta( const Dataset& data, const Data& added ) const = 0;
	
};

/*! \brief Calcualtes the Cauchy Schwarz divergence between a kernel density estimate
 * using the data versus a uniform distribution. Returns infinity for an empty dataset. */
template <class Data>
class UniformCauchySchwarzDivergence
: public DatasetFunction<Data>
{
public:

	typedef	std::shared_ptr<UniformCauchySchwarzDivergence> Ptr;
	typedef KernelFunction<Data> Kernel;
	
	typedef Data DataType;
	typedef std::vector<Data> Dataset;
	
	// Give the kernel used in the KDE
	UniformCauchySchwarzDivergence( typename Kernel::Ptr& k, double bandwidth = 1.0 ) 
	: kernel( k ), hr( 1.0 / bandwidth ) {}
	
	virtual double operator()( const Dataset& data ) const
	{
		if( data.empty() ) { return std::numeric_limits<double>::infinity(); }
		
		double acc = 0;
		for( unsigned int i = 0; i < data.size()-1; i++ )
		{
			for( unsigned int j = i+1; j < data.size(); j++ )
			{
				acc += Evaluate( data[i], data[j] );
			}
		}
		acc *= 2;
		
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			acc += Evaluate( data[i], data[i] );
		}
		
		return std::log( acc / ( data.size()*data.size() ) );
	}
	
	virtual double operator()( const Eigen::MatrixXf& K ) const
	{
		return std::log( K.sum()/( K.rows()*K.cols() ) );
	}
	
	virtual double delta( const Dataset& data, const Data& added ) const 
	{
		if( data.empty() ) { return -std::numeric_limits<double>::infinity(); }
		
		double acc = 0;
		for( unsigned int i = 0; i < data.size()-1; i++ )
		{
			for( unsigned int j = i+1; j < data.size(); j++ )
			{
				acc += Evaluate( data[i], data[j] );
			}
		}
		acc *= 2;
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			acc += Evaluate( data[i], data[i] );
		}
		
		// diagonal elements are 0
		unsigned int N = data.size();
		double origDCS = std::log( acc / ( N*N ) );
		
		double addedAcc = 0;
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			addedAcc += Evaluate( added, data[i] );
		}
		addedAcc *= 2;
		addedAcc += Evaluate( added, added );
		addedAcc += acc;
		
		N = N+1;
		double addedDCS = std::log( addedAcc / ( N*N ) );
		
		Dataset cop( data );
		cop.push_back( added );
		
		return addedDCS - origDCS;
	}

private:
	
	typename Kernel::Ptr kernel;
	double hr;
	
	double Evaluate( const Data& a, const Data& b ) const
	{
		return kernel->Evaluate( kernel->Difference( a, b ) * hr );
	}
	
};

} // end namespace calotypes
