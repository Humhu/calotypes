#pragma once

#include <boost/function.hpp>

namespace calotypes
{
	
// TODO Add multi-threading
/*! \brief A basic templatized cross validation framework. */
template <class Model, class Data>
class CrossValidator
{
public:
	
	typedef std::vector<Data> Dataset;
	typedef boost::function<Model(const Dataset&)> TrainFunc;
	typedef boost::function<double(const Model&, const Dataset&)> TestFunc;
	
	/*! \brief Create a cross validator with specified train and test function. */
	CrossValidator( TrainFunc train, TestFunc test )
		: trainer( train ), tester( test ) {}

	/*! \brief Performs numFolds-fold cross validation. Returns models, error
	 * for each fold, and the average error. */
	double PerformCrossValidation( const Dataset& data, unsigned int numFolds,
									std::vector<Model>& models,
									std::vector<double>& errors )
	{
		models.resize( numFolds );
		errors.resize( numFolds );
		
		double acc = 0;
		Dataset testSet, trainSet;
		for( unsigned int foldInd = 0; foldInd < numFolds; foldInd++ )
		{
			GetFold( data, numFolds, foldInd, testSet, trainSet );
			models[ foldInd ] = trainer( trainSet );
			errors[ foldInd ] = tester( models[ foldInd ], testSet );
			acc += errors[ foldInd ];
		}
		return acc / numFolds;
		
	}
	
	/*! \brief Splits a data into folds and returns the test and train set
		* corresponding to fold ind. */
	static void GetFold( const Dataset& data, unsigned int numFolds, unsigned int ind,
							Dataset& test, Dataset& train )
	{
		if( numFolds == 1 )
		{
			throw std::logic_error( "Cannot run cross validation with 1 fold!" );
		}
		
		double foldSize = data.size() / (double) numFolds;
		unsigned int foldStart = std::round( ind*foldSize );
		unsigned int foldEnd = std::round( (ind+1)*foldSize );
		
		// Copy test fold
		test.clear();
		test.resize( foldEnd - foldStart );
		std::copy( data.begin() + foldStart, data.begin() + foldEnd, test.begin() );
		
		train.clear();
		train.resize( data.size() - test.size() );
		unsigned int trainCount = 0;
		// Have to catch cases where the test fold is at the ends of the dataset
		if( ind > 0 )
		{
			std::copy( data.begin(), data.begin() + foldStart, train.begin() );
			trainCount += foldStart;
		}
		if( ind < numFolds - 1 )
		{
			std::copy( data.begin() + foldEnd, data.end(), train.begin() + trainCount );
		}
	}
	
private:
	
	TrainFunc trainer;
	TestFunc tester;
};
	
} // end namespace calotypes
