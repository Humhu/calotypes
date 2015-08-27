#pragma once

#include <boost/thread/thread.hpp>
#include <boost/function.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace calotypes
{
	
/*! \brief Splits a dataset into folds and returns the test and train set
	* corresponding to fold ind. */
template< class Data >
void GetFold( const std::vector<Data>& data, unsigned int numFolds, unsigned int ind,
			  std::vector<Data>& test, std::vector<Data>& train )
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
	
/*! \brief A basic templatized cross validation task representation. It is designed
 * to be operated on in parallel, but can work in serial as well. Note that it
 * takes a const reference to the dataset, so the data should not be changed afterwards. */
template <class Model, class Data>
class CrossValidationTask
{
public:
	
	typedef std::vector<Data> Dataset;
	
	/*! \brief Training functions take a dataset, assign a model, and return the
	 * training time error. They may also return just a subset of the training set. */
	typedef boost::function<double (Model&, Dataset&)> TrainFunc;
	
	/*! \brief Testing functions take a model and dataset and return the test
	 * time error. */
	typedef boost::function<double (const Model&, const Dataset&)> TestFunc;

	/*! \brief Results for validation of a single fold. */
	struct ValidationResult
	{
		Model model;
		unsigned int foldIndex;
		double trainingError;
		double testError;
		Dataset trainingData;
		boost::posix_time::time_duration trainingTime;
	};
	
	const TrainFunc trainer;
	const TestFunc tester;
	const Dataset& data; // We have to trust the data doesn't change!
	const unsigned int numFolds;

	CrossValidationTask( TrainFunc tr, TestFunc te, const Dataset& dat, unsigned int k )
	: trainer( tr ), tester( te ), data( dat ), numFolds( k ), results( k )
		{}
	
	std::vector<ValidationResult> GetResults()
	{
		boost::unique_lock< boost::mutex > lock( mutex );
		return results;
	}
	
	/*! \brief Performs cross-validation for all folds. */
	void Validate()
	{
		for( unsigned int i = 0; i < numFolds; i++ )
		{
			ValidateFold( i );
		}
	}
	
	/*! \brief Executes cross-validation for fold index foldInd. */
	void ValidateFold( unsigned int foldInd )
	{
		Dataset testSet, trainSet;
		Model model;
		double trainingError, testError;
		boost::posix_time::time_duration trainingTime;
		
		// NOTE Concurrent access to const functors should be safe...?
		GetFold( data, numFolds, foldInd, testSet, trainSet );
		boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
		trainingError = trainer( model, trainSet );
		boost::posix_time::ptime after = boost::posix_time::microsec_clock::local_time();
		trainingTime = after - now;
		testError = tester( model, testSet );
		
		boost::unique_lock<boost::mutex> lock( mutex );
		results[foldInd].model = model;
		results[foldInd].trainingError = trainingError;
		results[foldInd].testError = testError;
		results[foldInd].trainingTime = trainingTime;
		results[foldInd].trainingData = trainSet;
		return;
	}
	
private:
	
	boost::mutex mutex;
	std::vector<ValidationResult> results;
	
};
	
} // end namespace calotypes
