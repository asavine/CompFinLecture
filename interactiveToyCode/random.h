#pragma once

#include <vector>
#include <memory>
using namespace std;

//  Random number generators
//  ========================

class RNG
{
public:
    
    //  Initialise with dimension simDim
    virtual void init(const size_t simDim) = 0;

    //  Compute the next vector[simDim] of independent Uniforms or Gaussians
    //  The vector is filled by the function and must be pre-allocated
	virtual void nextU(vector<double>& uVec) = 0;
	virtual void nextG(vector<double>& gaussVec) = 0;

    virtual unique_ptr<RNG> clone() const = 0;

    virtual ~RNG() {}

    //  Skip ahead
    virtual void skipTo(const unsigned b) = 0;
};
