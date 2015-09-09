#pragma once

#include <memory>

namespace calotypes
{

/*! \brief Interface for PDFs that return probabilities of data. */
template <class Data>
class ProbabilityDensityFunction
{
public:
	
	typedef std::shared_ptr<ProbabilityDensityFunction> Ptr;
	
	ProbabilityDensityFunction() {}
	
	/*! \brief Evaluate the PDF at x. */
	virtual double operator()( const Data& x ) const = 0;
	
	/*! \brief Return whether the PDF is normalized or not. */
	virtual bool IsNormalized() const = 0;
	
	/*! \brief Returns the joint probability assuming independent samples. */
	virtual double operator()( const std::vector<Data>& data ) const
	{
		double acc = 1.0;
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			acc *= (*this)( data[i] );
		}
		return acc;
	}
	
	/*! \brief Returns the log probability of independent samples. */
	virtual double LogProbability( const std::vector<Data>& data ) const
	{
		double acc = 0;
		for( unsigned int i = 0; i < data.size(); i++ )
		{
			acc += std::log( (*this)( data[i] ) );
		}
		return acc;
	}
	
};

template <class Data>
class UniformDensityFunction
: public ProbabilityDensityFunction<Data>
{
public:
	
	typedef std::shared_ptr<UniformDensityFunction> Ptr;
	
	UniformDensityFunction( double c = 1.0 )
	: val( c ) {}
	
	virtual double operator()( const Data& x ) const { return val; }
	virtual bool IsNormalized() const { return false; }
	
private:
	
	double val;
};

}
