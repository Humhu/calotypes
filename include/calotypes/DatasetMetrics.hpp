#pragma once

#include <memory>
#include <Eigen/Dense>

namespace calotypes
{

/*! \brief Base interface for metrics that operate on pairs of data. */
template <class Data>
class DataMetric 
{
public:

	typedef std::shared_ptr<DataMetric> Ptr;
	typedef Data DataType;
	
	DataMetric() {}
	
	virtual double operator()( const Data& a, const Data& b ) const = 0;
	
};

/*! \brief Negates a metric. */
template <class Data>
class NegativeMetric
: public DataMetric<Data>
{
public:
	
	typedef std::shared_ptr<NegativeMetric> Ptr;
	
	NegativeMetric( DataMetric<Data>& m )
	: metric( m ) {}
	
	virtual double operator()( const Data& a, const Data& b ) const
	{
		return -metric( a, b );
	}
	
private:
	DataMetric<Data>& metric;
};

/*! \brief Computes the positive-definite kernel (distance) matrix for a given metric. 
 * Assumes metric(a,a) = 0 and metric(a,b) = metric(b,a). */
template <class Data>
void ComputeDistanceMatrix( const DataMetric<Data>& metric, const std::vector<Data>& data,
							const Eigen::MatrixXf& K )
{
	unsigned int N = data.size();
	K = Eigen::MatrixXf( N, N );
	for( unsigned int i = 1; i < N-1; i++ )
	{
		for( unsigned int j = i+1; j < N; j++ )
		{
			K(i,j) = metric( data[i], data[j] );
		}
	}
	
	for( unsigned int i = 0; i < N; i++ ) { K(i,i) = 0; }
	for( unsigned int j = 1; j < N-1; j++ )
	{
		for( unsigned int i = j+1; i < N; i++ )
		{
			K(i,j) = K(j,i);
		}
	}
}

/*! \brief Base interface for metrics that operate on sets of data and return scalars. */
template <class Data>
class DatasetMetric
{
public:
	
	typedef std::shared_ptr<DatasetMetric> Ptr;
	
	typedef Data DataType;
	typedef std::vector<Data> Dataset;
	
	DatasetMetric() {}
	
	/*! \brief Uses a precomputed kernel matrix. */
	virtual double operator()( const Eigen::MatrixXf& K ) const = 0;
	
	/*! \brief Calculates pairwise metrics in place. */
	virtual double operator()( const Dataset& data ) const = 0;
	
	/*! \brief Returns the change in metric associated with adding the specified data. */
	virtual double delta( const Dataset& data, const Data& added ) const = 0;
	
};

/*! \brief Computes the averaged mean pairwise metric over the dataset. */
template <class Data>
class AverageDatasetMetric
: public DatasetMetric<Data>
{
public:
	
	typedef std::shared_ptr<AverageDatasetMetric> Ptr;
	typedef typename DatasetMetric<Data>::Dataset Dataset;
	
	AverageDatasetMetric( const typename DataMetric<Data>::Ptr& m ) 
	: metric( m ) {}
	
	// TODO Some way to consolidate this and move the internal op out?
	virtual double operator()( const Eigen::MatrixXf& K ) const
	{
		unsigned int N = K.rows(); // TODO check square
		double acc = 0;
		for( unsigned int i = 0; i < N-1; i++ )
		{
			for( unsigned int j = i+1; j < N; j++ )
			{
				acc += K( i, j );
			}
		}
		return acc / ( N*N ); //
	}
	
	virtual double operator()( const Dataset& data )  const
	{
		unsigned int N = data.size();
		double acc = 0;
		for( unsigned int i = 0; i < N-1; i++ )
		{
			for( unsigned int j = i+1; j < N; j++ )
			{
				acc += (*metric)( data[i], data[j] );
			}
		}
		return acc / ( N*N );
	}
	
	virtual double delta( const Dataset& data, const Data& added ) const
	{
		unsigned int N = data.size();
		double acc = 0;
		for( unsigned int i = 0; i < N; i++ )
		{
			acc += (*metric)( data[i], added );
		}
		return acc / N;
	}
	
private:
	
	typename DataMetric<Data>::Ptr metric;
};

/*! \brief Computes the averaged max pairwise metric over the dataset. */
template <class Data>
class MaxDatasetMetric
: public DatasetMetric<Data>
{
public:

	typedef std::shared_ptr<MaxDatasetMetric> Ptr;
	typedef typename DatasetMetric<Data>::Dataset Dataset;

	MaxDatasetMetric( const typename DataMetric<Data>::Ptr& m )
	: metric( m ) {}
	
	virtual double operator()( const Eigen::MatrixXf& K ) const
	{
		unsigned int N = K.rows();
		double acc = 0;
		double best, curr;
		for( unsigned int i = 0; i < N; i++ )
		{
			best = -std::numeric_limits<double>::infinity();
			for( unsigned int j = 0; j < N; j++ )
			{
				if( j == i ) { continue; }
				curr = K(i,j);
				if( curr > best ) { best = curr; }
			}
			acc += best;
		}
		return acc / N;
	}
	
	virtual double operator()( const Dataset& data ) const
	{
		unsigned int N = data.size();
		double acc = 0;
		double best, curr;
		for( unsigned int i = 0; i < N; i++ )
		{
			best = -std::numeric_limits<double>::infinity();
			for( unsigned int j = 0; j < N; j++ )
			{
				if( j == i ) { continue; }
				curr = *metric( data[i], data[j] );
				if( curr > best ) { best = curr; }
			}
			acc += best;
		}
		return acc / N;
	}
	
	virtual double delta( const Dataset& data, const Data& added ) const
	{
		Dataset c( data );
		c.push_back( added );
		return operator()( c );
	}

private:
	
	typename DataMetric<Data>::Ptr metric;
};

} // end namespace calotypes
