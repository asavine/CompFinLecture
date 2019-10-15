#pragma once

#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>

#include "matrix.h"     //  simple matrix class
#include "interp.h"     //  interpolator
#include "gaussians.h"  //  normal distribution
#include "utility.h"    //  "filler"

#include <iostream>

//  Changes from first version (refactoring) to last (parallel AAD)
//  - Thread local tape on AAD.h and AAD.cpp
//  - init() and generatePaths() have V1 and V2 instances in Dupire
//  - Value driver calls parallel simulator instead of serial

/*  ************************************************************************************************ /
                                  Original Dupire code from toyAAD
/   *************************************************************************************************/

//  Code for AAD: record, custom number, operator overloading and adjoint propagation 
//  was moved to AAD.h, and the global tape was moved to the Number type as a static data member
#include "AAD.h"

class RNG;
template <class T>
inline T toyDupireBarrierMc(
    //  Spot
    const T            S0,
    //  Local volatility
    const vector<T>&   spots,
    const vector<T>&   times,
    const matrix<T>&   vols,
    //  Product parameters
    const T            maturity,
    const T            strike,
    const T            barrier,
    //  Number of paths and time steps
    const int          Np,
    const int          Nt,
    //  Smoothing
    const T            epsilon,
    //  Initialized random number generator
    RNG&               random)
{
    //  Initialize
    T result = 0;
    vector<double> gaussianIncrements(Nt); // double because the RNG is not templated (and doesn't need to be, see chapter 12)
    const T dt = maturity / Nt, sdt = sqrt(dt);

    //  Loop over paths
    for (int i = 0; i < Np; ++i)
    {
        //  Generate Nt Gaussian Numbers
        random.nextG(gaussianIncrements);
        //  Step by step
        T spot = S0, time = 0;
        /*  bool alive = true; */ T alive = 1.0; // alive is a real number in (0,1)
        for (size_t j = 0; j < Nt; ++j)
        {
            //  Interpolate volatility
            const T vol = interp2D(spots, times, vols, spot, time);
            time += dt;
            //  Simulate return
            spot *= exp(-0.5 * vol * vol * dt + vol * sdt * gaussianIncrements[j]);
            //  Monitor barrier
            /* if (spot > barrier) { alive = false; break; } */
            if (spot > barrier + epsilon) { alive = 0.0; break; }       //   definitely dead
            else if (spot < barrier - epsilon) { /* do nothing */ }     //   definitely alive
            else /* in between, interpolate */ alive *= 1.0 - (spot - barrier + epsilon) / (2 * epsilon);

        }
        //  Payoff
        /* if (alive && spot > strike) result += spot - strike; */ if (spot > strike) result += alive * (spot - strike); // pay on surviving notional
    }   //  paths

    return result / Np;
}

inline void dupireRisksMiniBatch(
	const double S0, const vector<double>& spots, const vector<double>& times, const matrix<double>& vols,
	const double maturity, const double strike, const double barrier,
	const int Np, const int Nt, const double epsilon, RNG& random,
	/* results: value and dV/dS, dV/d(local vols) */ double& price, double& delta, matrix<double>& vegas)
{

	//	1. Initialize inputs, record on tape

	Number nS0(S0), nMaturity(maturity), nStrike(strike), nBarrier(barrier), nEpsilon(epsilon);
	vector<Number> nSpots(spots.size()), nTimes(times.size());
	matrix<Number> nVols(vols.rows(), vols.cols());

	for (int i = 0; i < spots.size(); ++i) nSpots[i] = Number(spots[i]);
	for (int i = 0; i < times.size(); ++i) nTimes[i] = Number(times[i]);
	for (int i = 0; i < vols.rows(); ++i) for (int j = 0; j < vols.cols(); ++j) nVols[i][j] = Number(vols[i][j]);

	//	2. Call instrumented evaluation code, which evaluates the barrier option price and records all operations

	Number nPrice = toyDupireBarrierMc(nS0, nSpots, nTimes, nVols, nMaturity, nStrike, nBarrier, Np, Nt, nEpsilon, random);

	//	3. Adjoint propagation

    //  propagate adjoints
    vector<double> adjoints = calculateAdjoints(nPrice);

	//	4. Pick results

	price = nPrice.value;
	delta = adjoints[nS0.idx];
	for (int i = 0; i < vols.rows(); ++i) for (int j = 0; j < vols.cols(); ++j) vegas[i][j] = adjoints[nVols[i][j].idx];
}

inline void toyDupireBarrierMcRisks(
	const double S0, const vector<double>& spots, const vector<double>& times, const matrix<double>& vols,
	const double maturity, const double strike, const double barrier,
	const int Np, const int Nt, const double epsilon, RNG& random,
	/* results: value and dV/dS, dV/d(local vols) */ double& price, double& delta, matrix<double>& vegas)
{
	price = 0;
	delta = 0;
	for (int i = 0; i < vegas.rows(); ++i) for (int j = 0; j < vegas.cols(); ++j) vegas[i][j] = 0;

	double batchPrice, batchDelta;
	matrix<double> batchVegas(vegas.rows(), vegas.cols());

	int pathsToGo = Np, pathsPerBatch = 512;
	while (pathsToGo > 0)
	{
		//	wipe tape
		Number::tape.clear();

		//	do mini batch
		int paths = min(pathsToGo, pathsPerBatch);
		dupireRisksMiniBatch(S0, spots, times, vols, maturity, strike, barrier, paths, Nt, epsilon, random, batchPrice, batchDelta, batchVegas);

		//	update results
		price += batchPrice * paths / Np;
		delta += batchDelta * paths / Np;
		for (int i = 0; i < vegas.rows(); ++i) for (int j = 0; j < vegas.cols(); ++j) 
			vegas[i][j] += batchVegas[i][j] * paths / Np;

		pathsToGo -= paths;
	}
}

/*  ************************************************************************************************ /
                           Towards a scalable, generic simulation library
/   *************************************************************************************************/

/*  ************************************************************************************************ /
                                    Step 1: (simplistic) scenarios
/   *************************************************************************************************/

using Time = double;
extern Time systemTime;

template <class T>
struct Sample
{
    T       numeraire;
    T       spot;
};

template <class T>
using Scenario = vector<Sample<T>>;

/*  ************************************************************************************************ /
            Step 2: interface classes for random number generators, products and models
/   *************************************************************************************************/

//  2.1 Random number generators (RNG)
//  implemented in different files
#include "random.h"     
#include "sobol.h"
#include "mrg32k3a.h"

//  2.2 Products
//  A product's main responsibility is evaluate payoff on a given scenario

//  Note const correctness

template <class T>
class Product
{
public:

    //  Access to the product timeline = collection of event dates
    virtual const vector<Time>& timeline() const = 0;

    //  Compute payoff given a scenario
    virtual T payoff(const Scenario<T>& path) const = 0;

    //  Virtual copy constructor idiom
    virtual unique_ptr<Product<T>> clone() const = 0;

    //  Hierarchical patterns must define virtual destructors, even if they do nothing
    virtual ~Product() {}
};

//  2.3 Models
//  A model's main reponsibility is to generate scenarios

//  Implementation must allow to manipulate heterogeneous parameters consistently across models

template <class T>
class Model
{
protected:

    //  ----- generically deal with parameters -----

    //  General set of hyperparameters
    unordered_map<string, matrix<double>>   myHyperParams;

    //  General set of model parameters
    unordered_map<string, matrix<double>>   myParams;

public:

    //  Set generic hyperparameters and parameters
    void setHyperParams(const unordered_map<string, matrix<double>>& hp) { myHyperParams = hp; resetHyperParams(); }
    void setParams(const unordered_map<string, matrix<double>>& p) { myParams = p; resetParams(); }

    //  Get
    const unordered_map<string, matrix<double>>& hyperparams() const { return myHyperParams; }
    const unordered_map<string, matrix<double>>& params() const { return myParams; }
    virtual const matrix<T>& param(const string& whatParam) const = 0;

    //  (Re-)set on model
    virtual void resetHyperParams() = 0;
    virtual void resetParams() = 0;

    //  ----- end parameters -----

    //  Initialize: get ready for simulation, given a timeline (from the product)
    virtual void init(const vector<Time>& prdTimeline) = 0;

    //  Access to the MC dimension = number of random numbers for one path
    virtual size_t simDim() const = 0;

    //  Generate a path consuming a vector[simDim()] of independent Gaussians
    //  return results in a pre-allocated scenario
    virtual void generatePath(
        const vector<double>&       gaussVec, 
        Scenario<T>&                path) 
            const = 0;

    //  Virtual copy constructor idiom
    virtual unique_ptr<Model<T>> clone() const = 0;

    //  Hierarchical patterns must define virtual destructors, even if they do nothing
    virtual ~Model() {}
};

/*  ************************************************************************************************ /
                                Step 3: template simulation algorithm
/   *************************************************************************************************/

//  Returns vector of payoffs in different scenarios
inline matrix<double> mcSimul(
    const Product<double>&      prd,
    const Model<double>&        mdl,
    const RNG&                  rng,			        
    const size_t                nBatch,
    const size_t                nPath)                      
{
    //	Allocate results
    matrix<double> results(nBatch, nPath);

    //  Work with copies of the model and RNG
    //      which are modified when we set up the simulation
    //  Copies are OK at high level
    auto cMdl = mdl.clone();
    auto cRng = rng.clone();

    //  Prepare model for simulation, given the product's timeline
    cMdl->init(prd.timeline());

    //  Init the RNG, given the MC dimension (number of random numbers for one path), given by model
    cRng->init(cMdl->simDim());                        
    
    //  Allocate space for Gaussian numbers
    vector<double> gaussVec(cMdl->simDim());

    //  Allocate space for path
    Scenario<double> path(prd.timeline().size());

    //	Iterate through paths	
    for (size_t i = 0; i<nBatch; ++i)
    {
        for (size_t j = 0; j<nPath; ++j)
        {
            //  Next Gaussian vector, dimension D
            cRng->nextG(gaussVec);                        
            //  Generate path, consuming the Gaussian vector
            cMdl->generatePath(gaussVec, path);     
            //	Compute result = payoff over path
            results[i][j] = prd.payoff(path);
        }    
    }

    return results;	//	C++11: move
}

/*  ************************************************************************************************ /
                                    Step 4: concrete products 
/   *************************************************************************************************/

//  We now have the structure we need to define the up and out call product

template <class T>
class UOC : public Product<T>
{
    double              myStrike;
    double              myBarrier;
    Time                myMaturity;
    
    double              mySmooth;
    
    vector<Time>        myTimeline;

public:

    //  Constructor: store member data and build timeline
    //  Timeline = system date to maturity, with steps every monitoring frequency
    UOC(const double    strike,
        const double    barrier,
        const Time      maturity,
        const Time      monitorFreq,
        const double    smooth)
        : myStrike(strike),
        myBarrier(barrier),
        myMaturity(maturity),
        mySmooth(smooth)
    {
        //  Timeline

        //  Today
        myTimeline.push_back(systemTime);
        Time t = systemTime + monitorFreq;

        //  Barrier monitoring
        while (myMaturity > t + 1.0e-4)
        {
            myTimeline.push_back(t);
            t += monitorFreq;
        }

        //  Maturity
        myTimeline.push_back(myMaturity);
    }

    //  Virtual copy constructor
    unique_ptr<Product<T>> clone() const override
    {
        return make_unique<UOC<T>>(*this);
    }

    //  Timeline
    const vector<Time>& timeline() const override
    {
        return myTimeline;
    }

    //  Payoff
    T payoff(const Scenario<T>& path) const override
    {
        //  Monitor barrier
        T alive = 1.0;
        for (const Sample<T>& sample: path)
        {
            if (sample.spot > myBarrier + mySmooth) { alive = 0.0; break; }         //   definitely dead
            else if (sample.spot < myBarrier - mySmooth) { /* do nothing */ }       //   definitely alive
            else /* in between, interpolate */ alive *= 1.0 - (sample.spot - myBarrier + mySmooth) / (2 * mySmooth);
        }
        //  Payoff
        return alive * max(path.back().spot - myStrike, T(0.0)) / path.back().numeraire;
    }
};

/*  ************************************************************************************************ /
                                    Step 5: concrete models
/   *************************************************************************************************/

//  We now have the structure we need to define the Dupire model

template <class T>
class Dupire : public Model<T>
{
    //  Hyperparameters

    //  Local volatility structure
    vector<double>          mySpots;    
    vector<Time>            myTimes;    
    //  Maximum space between time steps
    Time                    myMaxDt;

    //  Parameters, all matrices

    //  Today's spot
    //  Note all parameters are matrices, so spot is a matrix of shape 1 x 1
    matrix<T>               mySpot;
    //  Local vols
    //  Spot major: sigma(spot i, time j) = myVols[i][j]
    matrix<T>               myVols;

    //  We assume interest rates, dividends, repo are all zero

    //  Workspace

    //  Similuation timeline
    vector<Time>            myTimeline;
    //  true (1) if the time step is an event date
    //  false (0) if it is an additional simulation step
    vector<bool>            myCommonSteps;

public:

    //  Constructor
    Dupire(const unordered_map<string, matrix<double>>& hyperparams, const unordered_map<string, matrix<double>>& params)
    {
        //  Always set hyperparams first
        setHyperParams(hyperparams);
        setParams(params);
    }

    //  Deal with hyperparameters and parameters
    
    //  Get
    const matrix<T>& param(const string& whatParam) const override
    {
        if (whatParam == "spot") return mySpot;
        else if (whatParam == "vols") return myVols;
        else throw runtime_error("unknown parameter requested");
    }

    //  Set

    //  Hyperparams
    void resetHyperParams() override
    {
        const matrix<double>& spots = myHyperParams.at("spots");
        mySpots.resize(spots.rows());
        copy(spots.begin(), spots.end(), mySpots.begin());
        
        const matrix<double>& times = myHyperParams.at("times");
        myTimes.resize(times.rows());
        copy(times.begin(), times.end(), myTimes.begin());

        const matrix<double>& maxDt = myHyperParams.at("maxDt");
        myMaxDt = maxDt[0][0];

        //  Size/allocate params
        mySpot.resize(1, 1);
        myVols.resize(mySpots.size(), myTimes.size());
    }

    //  Params
    void resetParams() override
    {
        mySpot[0][0] = myParams.at("spot")[0][0];
        
        const auto& vols = myParams.at("vols");
        copy(vols.begin(), vols.end(), myVols.begin());
    }

    //  Virtual copy constructor
    unique_ptr<Model<T>> clone() const override
    {
        auto clone = make_unique<Dupire<T>>(*this);
        return clone;
    }

    //  MC Dimension
    //  = Number of random numbers for one path
    //  = Time steps (intervals) on the simulation timeline (not the product timeline!)
    size_t simDim() const override
    {
        return myTimeline.size() - 1;
    }

    void init(const vector<Time>& prdTimeline) override
    {
        initV1(prdTimeline);
    }

    void generatePath(
    const vector<double>&       gaussVec, 
    Scenario<T>&                path) 
        const override
    {
        generatePathV1(gaussVec, path);
    }

    /*
        First version: determine timeline in int(), generate in generatePath()
    */

    //  Initialize: get ready for simulation, given a timeline (from the product)
    void initV1(const vector<Time>& prdTimeline) 
    {
        //  Here, we set the simulation timeline
        //  We have the product's timeline
        //  And we add 
        //      - today, i.e. the "system time" 
        //      - additional time steps so step size never exceeds max Dt

        //  We use a boilerplate "filling" routine coded in utility.h
        myTimeline = fillData(
            prdTimeline,        // Original (product) timeline
            myMaxDt,            // Maximum step size
            1.0e-03,            // Minimum step size
            &systemTime, &systemTime + 1);  //  Hack to include system time

        //  Mark steps on timeline that were on the original product timeline
        myCommonSteps.resize(myTimeline.size());
        transform(myTimeline.begin(), myTimeline.end(), myCommonSteps.begin(), 
            [&](const Time t)
        {
            return binary_search(prdTimeline.begin(), prdTimeline.end(), t);
        });
    }

    //  Generate a path consuming a vector[simDim()] of independent Gaussians
    //  return results in a pre-allocated scenario

    //  Note, here we move the relevant part of the original Dupire MC code
    void generatePathV1(
        const vector<double>&       gaussVec, 
        Scenario<T>&                path) 
            const 
    {
        const size_t time_steps = myTimeline.size();

        //  Today
        T spot = mySpot[0][0], time = 0;
        
        //  Careful: scenario is on the product timeline, not the simulation timeline
        size_t scen_idx = 0;
        if (myCommonSteps[0])
        {
            path[scen_idx].spot = spot;
            path[scen_idx].numeraire = 1;
            ++scen_idx;
        }
        
        for (size_t step = 1; step < time_steps; ++step)
        {
            //  Interpolate volatility
            const T vol = interp2D(mySpots, myTimes, myVols, spot, time);
            const double dt = myTimeline[step] - myTimeline[step - 1], sdt = sqrt(dt);

            //  Simulate return
            spot *= exp(-0.5 * vol * vol * dt + vol * sdt * gaussVec[step - 1]);
            time = myTimeline[step];

            //  On scenario (i.e. product) timeline?
            if (myCommonSteps[step])
            {
                path[scen_idx].spot = spot;
                path[scen_idx].numeraire = 1;
                ++scen_idx;
            }
        }
    }

    /*
        Better version: put as much work as possible in init()
    */

    private:

    //  Store vols here, pre-interpolated in time, in time major, i.e. sigma(spot i, time j) = myInterpVols[j][i]
    matrix<T>               myInterpVols;

    public:

    //  Initialize: get ready for simulation, given a timeline (from the product)
    void initV2(const vector<Time>& prdTimeline) 
    {
        myTimeline = fillData(
            prdTimeline,        // Original (product) timeline
            myMaxDt,            // Maximum step size
            1.0e-03,            // Minimum step size
            &systemTime, &systemTime + 1);  //  Hack to include system time

        myCommonSteps.resize(myTimeline.size());
        transform(myTimeline.begin(), myTimeline.end(), myCommonSteps.begin(), 
            [&](const Time t)
        {
            return binary_search(prdTimeline.begin(), prdTimeline.end(), t);
        });

        //  Preinterpolate and pre-multiply by sqrt(dt)
        //  +++ this is the only change +++
        const size_t n = myTimeline.size() - 1;
        const size_t m = mySpots.size();
        myInterpVols.resize(n, m);
        for (size_t i = 0; i < n; ++i)
        {
            const double sqrtdt = sqrt(myTimeline[i + 1] - myTimeline[i]);
            for (size_t j = 0; j < m; ++j)
            {
                myInterpVols[i][j] = sqrtdt * interp(
                    myTimes.begin(),
                    myTimes.end(),
                    myVols[j],
                    myVols[j] + myTimes.size(),
                    myTimeline[i]);
            }
        }
    }

    //  Generate a path consuming a vector[simDim()] of independent Gaussians
    //  return results in a pre-allocated scenario

    //  Note, here we move the relevant part of the original Dupire MC code
    void generatePathV2(
        const vector<double>&       gaussVec, 
        Scenario<T>&                path) 
            const 
    {
        const size_t time_steps = myTimeline.size();
        const size_t m = mySpots.size();

        T spot = mySpot[0][0], time = 0;
        
        size_t scen_idx = 0;
        if (myCommonSteps[0])
        {
            path[scen_idx].spot = spot;
            path[scen_idx].numeraire = 1;
            ++scen_idx;
        }
        
        for (size_t step = 1; step < time_steps; ++step)
        {

            //  +++ changes here +++

            //  Interpolate volatility in spot, reuse pre-interpolation in time
            const T vol = interp(mySpots.begin(), mySpots.end(), myInterpVols[step- 1], myInterpVols[step - 1] + m, spot);

            //  Simulate return, no need to multiply by sqrt(dt)
            spot *= exp(vol * (-0.5 * vol + gaussVec[step - 1]));
            time = myTimeline[step];

            //  +++ no more change +++

            if (myCommonSteps[step])
            {
                path[scen_idx].spot = spot;
                path[scen_idx].numeraire = 1;
                ++scen_idx;
            }
        }
    }
};

/*  ************************************************************************************************ /
                                            Parallel version
/   *************************************************************************************************/

//  Returns vector of payoffs in different scenarios
inline matrix<double> mcSimulParallel(
    const Product<double>&      prd,
    const Model<double>&        mdl,
    const RNG&                  rng,			        
    const size_t                nBatch,
    const size_t                nPath)                      
{
    matrix<double> results(nBatch, nPath);

    auto cMdl = mdl.clone();
    auto cRng = rng.clone();
    cMdl->init(prd.timeline());
    cRng->init(cMdl->simDim());                        

    //  Do not initialize global instances any more, instead, initialize local instances of mutable objects, inside the parallel loop
    //  vector<double> gaussVec(cMdl->simDim());
    //  Scenario<double> path(prd.timeline().size());

    //  openMp directive for parallelism here
    #pragma omp parallel for
    for (int i = 0; i<nBatch; ++i)  //  has to be an int, openMp refuses counters of type size_t
    {
        //  This is the parallel block

        //  Thread local objects
        //  These are ~mutable~ objects, which state is ~mutated~ inside the parallel loop
        //  Multiple threads cannot mutate the same objects, so each thread must own its own copy
        //  To make this happen, we create them ~inside~ the parallel loop
        //  (We wouldn't want to re-create them for every path, so batching comes in handy)
        auto local_rng = cRng->clone();
        vector<double> local_gaussVec(cMdl->simDim());
        Scenario<double> local_path(prd.timeline().size());
        //

        //  This is a crucial new step: set rng to the state for the generation of the (i * nPath)-th random point
        local_rng->skipTo(i * nPath);

        //  The inner loop doesn't change: process the nPath paths in batch i
        for (size_t j = 0; j<nPath; ++j)
        {
            local_rng->nextG(local_gaussVec);                              
            cMdl->generatePath(local_gaussVec, local_path);     
            results[i][j] = prd.payoff(local_path);
        }    
    }

    return results;	//	C++11: move
}

/*  ************************************************************************************************ /
                                         Parallel AAD version
/   *************************************************************************************************/

//  Returns vector of values and risks in different batches
//  For each batch, the result is a hash-map of strings to matrices, for example
//  result[b]["vol"][i][j] is the sensitivity of the value (average in batch b), to vol[i][j]
//  note we add an entry result[b]["value"] for the value
inline vector<unordered_map<string, matrix<double>>> mcSimulParallelAAD(
    const Product<Number>&      prd,
    const Model<Number>&        mdl,
    const RNG&                  rng,			        
    const size_t                nBatch,
    const size_t                nPath)                      
{
    //  vector of parameters
    vector<string> paramLabels;
    for (auto& kv : mdl.params()) paramLabels.push_back(kv.first);
    size_t nParams = paramLabels.size();

    //  prototype results: map of parameters to matrix
    unordered_map<string, matrix<double>> results;
    for (auto& kv : mdl.params())
    {
        auto& mat = results[kv.first];
        mat.resize(kv.second.rows(), kv.second.cols());
        fill(mat.begin(), mat.end(), 0.0);
    }
    //  actual results
    vector<unordered_map<string, matrix<double>>> all_results(nBatch);

    //  add an entry for value
    auto& mat = results["value"];
    mat.resize(1, 1);
    mat[0][0] = 0.0;

    auto cMdl = mdl.clone();
    auto cRng = rng.clone();
    cMdl->init(prd.timeline());
    cRng->init(cMdl->simDim());                        

    #pragma omp parallel for
    for (int i = 0; i<nBatch; ++i)  
    {
        auto local_rng = cRng->clone();

        vector<double> local_gaussVec(cMdl->simDim());
        Scenario<Number> local_path(prd.timeline().size());

        local_rng->skipTo(i * nPath);

        //  Model is now mutable (!!) due to adjoints of parameters
        auto local_mdl = cMdl->clone();

        //  Local results
        auto& local_results = all_results[i];
        local_results = results;    //  allocate and initialize

        //  Clear tape, recall it is local to each thread
        Number::tape.clear();
        
        //  Put parameters on tape
        local_mdl->resetParams();

        //  Must reinit model, so initialization is also on tape
        local_mdl->init(prd.timeline());

        //  Compute
        Number result = 0.0;
        for (size_t j = 0; j<nPath; ++j)
        {
            local_rng->nextG(local_gaussVec);                              
            local_mdl->generatePath(local_gaussVec, local_path);     
            result += prd.payoff(local_path);
        }    
        result /= nPath;

        //  Adjoints
        vector<double> adjoints = calculateAdjoints(result);
        
        //  Pick results
        local_results["value"][0][0] = result.value;
        for (size_t p=0; p<nParams; ++p)
        {
            auto& resmat = local_results[paramLabels[p]];
            auto& aadmat = local_mdl->param(paramLabels[p]);
            transform(aadmat.begin(), aadmat.end(), resmat.begin(), [&](const Number& n) { return adjoints[n.idx]; });
        }
    }

    return all_results;	//	C++11: move
}

/*  ************************************************************************************************ /
          Driver functions for a barrier option in Dupire's model in the generic framework
/   *************************************************************************************************/

inline double toyDupireBarrierMcV2(
    //  Spot
    const double            S0,
    //  Local volatility
    const vector<double>&   spots,
    const vector<double>&   times,
    const matrix<double>&   vols,
    //  Product parameters
    const double            maturity,
    const double            strike,
    const double            barrier,
    const double            monitoring,
    //  Number of paths and time steps
    const int               Nb,
    const int               Np,
    const double            maxDt,
    //  Smoothing
    const double            epsilon,
    //  Initialized random number generator
    RNG&                    random)
{
    //  Initialize the product
    UOC<double> uoc(strike, barrier, maturity, monitoring, epsilon);

    //  Ininitialize the model

    unordered_map<string, matrix<double>> hyperparams, params;
    
    //  hyper

    auto& pspots = hyperparams["spots"];
    pspots.resize(spots.size(), 1);
    copy(spots.begin(), spots.end(), pspots.begin());

    auto& ptimes = hyperparams["times"];
    ptimes.resize(times.size(), 1);
    copy(times.begin(), times.end(), ptimes.begin());

    //  We set max dt to the barrier monitoring frequency
    auto& pmaxdt = hyperparams["maxDt"];
    pmaxdt.resize(1, 1);
    pmaxdt[0][0] = maxDt;

    //  params

    auto& pspot = params["spot"];
    pspot.resize(1, 1);
    pspot[0][0] = S0;

    params["vols"] = vols;

    Dupire<double> dupire(hyperparams, params);
    dupire.init(uoc.timeline());

    //  Initialize the random generator
    random.init(dupire.simDim());

    //  Simulate
    matrix<double> simulations = mcSimul(uoc, dupire, random, Nb, Np);

    //  Average
    return accumulate(simulations.begin(), simulations.end(), 0.0) / (Nb * Np);
}

inline void toyDupireBarrierMcRisksV2(
	const double S0, 
    const vector<double>& spots, 
    const vector<double>& times, 
    const matrix<double>& vols,
	const double maturity, 
    const double strike, 
    const double barrier,
    const double monitoring,
	const int Nb, 
    const int Np, 
    const double maxDt, 
    const double epsilon, 
    RNG& random,
	/* results: value and dV/dS, dV/d(local vols) */ 
    double& price, 
    double& delta, 
    matrix<double>& vegas)
{
    //  Initialize the product
    UOC<Number> uoc(strike, barrier, maturity, monitoring, epsilon);

    //  Ininitialize the model

    unordered_map<string, matrix<double>> hyperparams, params;
    
    //  hyper

    auto& pspots = hyperparams["spots"];
    pspots.resize(spots.size(), 1);
    copy(spots.begin(), spots.end(), pspots.begin());

    auto& ptimes = hyperparams["times"];
    ptimes.resize(times.size(), 1);
    copy(times.begin(), times.end(), ptimes.begin());

    //  We set max dt to the barrier monitoring frequency
    auto& pmaxdt = hyperparams["maxDt"];
    pmaxdt.resize(1, 1);
    pmaxdt[0][0] = maxDt;

    //  params

    auto& pspot = params["spot"];
    pspot.resize(1, 1);
    pspot[0][0] = S0;

    params["vols"] = vols;

    Dupire<Number> dupire(hyperparams, params);
    dupire.init(uoc.timeline());

    //  Initialize the random generator
    random.init(dupire.simDim());

    //  Simulate
    auto batch_results = mcSimulParallelAAD(uoc, dupire, random, Nb, Np);

    //  Average across batches
    price = accumulate(batch_results.begin(), batch_results.end(), 0.0, 
        [](const double acc, const unordered_map<string, matrix<double>>& map4batch) { return acc + map4batch.at("value")[0][0]; })
        / Nb;
    delta = accumulate(batch_results.begin(), batch_results.end(), 0.0, 
        [](const double acc, const unordered_map<string, matrix<double>>& map4batch) { return acc + map4batch.at("spot")[0][0]; })
        / Nb;

    vegas.resize(vols.rows(), vols.cols());
    fill(vegas.begin(), vegas.end(), 0.0);
    for (size_t b=0; b<Nb; ++b)
    {
        transform(vegas.begin(), vegas.end(), batch_results[b]["vols"].begin(), vegas.begin(), 
            [](const double acc_vega, const double batch_vega) { return acc_vega + batch_vega; });
    }
    for (auto& vega : vegas) vega /= Nb;
}
