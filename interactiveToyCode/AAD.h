#pragma once

#include <vector>
using namespace std;

#include "gaussians.h"

/*  ************************************************************************************************ /
                                            Toy AAD code
/   *************************************************************************************************/

struct Record
{
    int     numArg;       //  Number of arguments: 0, 1 or 2
    int     idx1;         //  index of first argument on tape
    int     idx2;         //  index of second argument on tape
    double  der1;         //  partial derivative to first argument
    double  der2;         //  partial derivative to second argument
};

//  Custom Number type
struct Number
{
    //  The global tape, moved to the Number type as a static data member
    //  Thread local so different threads manipulate their own copy of the tape
    static thread_local vector<Record> tape;

    double  value;
    int     idx;

    //  default constructor does nothing
    Number() {}

    //  constructs with a value and record
    Number(const double& x) : value(x)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  reference record on tape
        idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 0;
    }

    Number operator +() const { return *this; }
    Number operator -() const { return Number(0.0) - *this; }

    Number& operator +=(const Number& rhs) { *this = *this + rhs; return *this; }
    Number& operator -=(const Number& rhs) { *this = *this - rhs; return *this; }
    Number& operator *=(const Number& rhs) { *this = *this * rhs; return *this; }
    Number& operator /=(const Number& rhs) { *this = *this / rhs; return *this; }

    friend Number operator+(const Number& lhs, const Number& rhs)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = lhs.value + rhs.value;

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 2;
        rec.idx1 = lhs.idx;
        rec.idx2 = rhs.idx;

        //  compute derivatives
        rec.der1 = 1;
        rec.der2 = 1;

        return result;
    }

    friend Number operator-(const Number& lhs, const Number& rhs)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = lhs.value - rhs.value;

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 2;
        rec.idx1 = lhs.idx;
        rec.idx2 = rhs.idx;

        //  compute derivatives -
        rec.der1 = 1;
        rec.der2 = -1;

        return result;
    }

    friend Number operator*(const Number& lhs, const Number& rhs)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = lhs.value * rhs.value;

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 2;
        rec.idx1 = lhs.idx;
        rec.idx2 = rhs.idx;

        //  compute derivatives *
        rec.der1 = rhs.value;
        rec.der2 = lhs.value;

        return result;
    }

    friend Number operator/(const Number& lhs, const Number& rhs)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = lhs.value / rhs.value;

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 2;
        rec.idx1 = lhs.idx;
        rec.idx2 = rhs.idx;

        //  compute derivatives /
        rec.der1 = 1.0 / rhs.value;
        rec.der2 = -lhs.value / (rhs.value * rhs.value);

        return result;
    }

    friend Number log(const Number& arg)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = log(arg.value);

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 1;
        rec.idx1 = arg.idx;

        //  compute derivative
        rec.der1 = 1.0 / arg.value;

        return result;
    }

    friend Number exp(const Number& arg)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = exp(arg.value);

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 1;
        rec.idx1 = arg.idx;

        //  compute derivative
        rec.der1 = result.value;

        return result;
    }

    friend Number sqrt(const Number& arg)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = sqrt(arg.value);

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 1;
        rec.idx1 = arg.idx;

        //  compute derivative
        rec.der1 = 0.5 / result.value;

        return result;
    }

    friend Number normalDens(const Number& arg)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = normalDens(arg.value);

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 1;
        rec.idx1 = arg.idx;

        //  compute derivative
        rec.der1 = -result.value * arg.value;

        return result;
    }

    friend Number normalCdf(const Number& arg)
    {
        //  create a new record on tape
        tape.push_back(Record());
        Record& rec = tape.back();

        //  compute result
        Number result;
        result.value = normalCdf(arg.value);

        //  reference record on tape
        result.idx = tape.size() - 1;

        //  populate record on tape
        rec.numArg = 1;
        rec.idx1 = arg.idx;

        //  compute derivative
        rec.der1 = normalDens(arg.value);

        return result;
    }

    friend bool operator==(const Number& lhs, const Number& rhs) { return lhs.value == rhs.value; }
    friend bool operator!=(const Number& lhs, const Number& rhs) { return lhs.value != rhs.value; }
    friend bool operator>(const Number& lhs, const Number& rhs) { return lhs.value > rhs.value; }
    friend bool operator>=(const Number& lhs, const Number& rhs) { return lhs.value >= rhs.value; }
    friend bool operator<(const Number& lhs, const Number& rhs) { return lhs.value < rhs.value; }
    friend bool operator<=(const Number& lhs, const Number& rhs) { return lhs.value <= rhs.value; }

};

inline vector<double> calculateAdjoints(Number& result)
{
    //  initialization
    vector<double> adjoints(Number::tape.size(), 0.0);  //  initialize all to 0
    int N = result.idx;                         //  find N
    adjoints[N] = 1.0;                          //  seed aN = 1
    
    //  backward propagation
    for(int j=N; j>0; --j)  //  iterate backwards over tape
    {
        if (Number::tape[j].numArg > 0)
        {
            adjoints[Number::tape[j].idx1] += adjoints[j] * Number::tape[j].der1;       //  propagate first argument

            if (Number::tape[j].numArg > 1)
            {
                adjoints[Number::tape[j].idx2] += adjoints[j] * Number::tape[j].der2;   //  propagate second argument
            }
        }
    }

    return adjoints;
}

