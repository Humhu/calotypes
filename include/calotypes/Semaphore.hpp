#pragma once

#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>

namespace calotypes 
{

/*! \brief A basic semaphore implementation. */
class Semaphore 
{
public:

	Semaphore( int startCounter = 0 )
	: counter( startCounter )
	{}
	
	void Increment( int i = 1 )
	{
		boost::unique_lock<Mutex> lock( mutex );
		counter += i;
		hasCounters.notify_all();
	}
	
	void Decrement( int i = 1 )
	{
		boost::unique_lock<Mutex> lock( mutex );
		
		// We will never reduce the counter below 0
		while( counter < i )
		{
			hasCounters.wait( lock );
		}
		counter = counter - i;
	}

	// TODO IncrementWait and DecrementWait
	// TODO Asserts to verify semaphore is never negative
	
protected:

	typedef boost::shared_mutex Mutex;
	typedef boost::condition_variable_any Condition;
	
	mutable Mutex mutex;
	int counter;
	Condition hasCounters;
	
};

} // end namespace calotypes
