#pragma once

#include <boost/function.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace calotypes
{
	
// TODO Add multi-threading
/*! \brief A basic templatized cross validation framework. */
template <class Model, class Data>
class CrossValidator
{
public:
	
	typedef std::vector<Data> Dataset;
	
	struct ValidationResult
	{
		Model model;
		unsigned int foldIndex;
		double trainingError;
		double testError;
		boost::posix_time::time_duration trainingTime;
	};
	
	/*! \brief Training functions take a dataset, assign a model, and return the
	 * training time error. */
	typedef boost::function<double (Model&, const Dataset&)> TrainFunc;
	
	/*! \brief Testing functions take a model and dataset and return the test
	 * time error. */
	typedef boost::function<double (const Model&, const Dataset&)> TestFunc;
	
	/*! \brief Create a cross validator with specified train and test function. */
	CrossValidator( TrainFunc train, TestFunc test )
		: trainer( train ), tester( test ) {}

	/*! \brief Performs numFolds-fold cross validation. Returns models, error
	 * for each fold, and the average error. */
	double CrossValidate( const Dataset& data, unsigned int numFolds,
								   std::vector<ValidationResult>& results )
	{
		results.resize( numFolds );
		
		double acc = 0;
		for( unsigned int foldInd = 0; foldInd < numFolds; foldInd++ )
		{
			PerformCrossValidation( data, numFolds, foldInd, results[foldInd] );
			acc += results[ foldInd ].testError;
		}
		return acc / numFolds;
		
	}
	
	void PerformCrossValidation( const Dataset& data, unsigned int numFolds,
								 unsigned int foldInd, ValidationResult& result )
	{
		Dataset testSet, trainSet;
		GetFold( data, numFolds, foldInd, testSet, trainSet );
		boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
		result.trainingError = trainer( result.model, trainSet );
		boost::posix_time::ptime after = boost::posix_time::microsec_clock::local_time();
		result.trainingTime = after - now;
		result.testError = tester( result.model, testSet );
		return;
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
