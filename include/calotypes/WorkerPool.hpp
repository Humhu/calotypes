#pragma once

#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>

#include <queue>

namespace calotypes 
{

/*! \brief An asynchronous worker thread pool. */
class WorkerPool 
{
public:

	typedef std::shared_ptr<WorkerPool> Ptr;
	typedef boost::function<void()> Job;
		
	WorkerPool( unsigned int n = 4 )
	: numWorkers( n ), numActive( 0 ) {}
		
	~WorkerPool()
	{
		StopWorkers();
	}
	
	void SetNumWorkers( unsigned int n )
	{
		numWorkers = n;
	}
	
	void EnqueueJob( Job job )
	{
		jobQueue.push( job );
		hasJobs.notify_one();
	}

	void StartWorkers()
	{
		for( unsigned int i = 0; i < numWorkers; i++) {
			workerThreads.create_thread( boost::bind( &WorkerPool::WorkerLoop, this ) );
		}
	}
	
	void StopWorkers()
	{
		workerThreads.interrupt_all();
		workerThreads.join_all();
	}

	/*! \brief Waits until job queue is empty and no threads are active. */
	void WaitOnJobs()
	{
		Lock lock( mutex );
		while( numActive > 0 || !jobQueue.empty() )
		{
			allThreadsDone.wait( lock );
		}
	}
	
protected:

	typedef boost::unique_lock< boost::shared_mutex > Lock;

	boost::shared_mutex mutex;
	unsigned int numWorkers;
	std::queue<Job> jobQueue;
	boost::condition_variable_any hasJobs;
	boost::condition_variable_any allThreadsDone;
	unsigned int numActive;
	boost::thread_group workerThreads;
	
	void WorkerLoop()
	{
		try 
		{
			while( true ) 
			{
				boost::this_thread::interruption_point();
	
				Lock lock( mutex );
				while( jobQueue.empty() )
				{
					hasJobs.wait( lock );
				}
				
				Job job = jobQueue.front();
				jobQueue.pop();
				numActive++;
				
				lock.unlock(); // Make sure we unlock or else this becomes serial!
				// TODO Catch exceptions in job?
				job();
				lock.lock();
				
				numActive--;
				if( numActive == 0 )
				{
					allThreadsDone.notify_all();
				}
				
			}
		}
		catch( boost::thread_interrupted e ) { return; }
	}

};

} // end namespace calotypes
