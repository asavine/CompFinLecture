#pragma once

#include <cmath>

template <class T>
bool equals(const T& x, const T& y)
{
    return x == y;
}

template <>
bool equals<double>(const double& x, const double& y)
{
    return fabs(x - y) < 1.0e-12;
}

//  simple matrix class, essentially a matrix view over a contiguous memory where the data is stored by row
template <class T>
class ptrMatrix
{
    size_t      myRows;
    size_t      myCols;
    T*          myVector;

public:

    typedef T value_type;

    //  Constructors
    ptrMatrix() : myRows(0), myCols(0), myVector(nullptr) {}
    ptrMatrix(const size_t rows, const size_t cols) : myRows(rows), myCols(cols), myVector(new T[rows*cols]) {}

    //  Copy, assign
    ptrMatrix(const ptrMatrix& rhs) : myRows(rhs.myRows), myCols(rhs.myCols), myVector(new T[myRows*myCols])
    {
        copy(rhs.myVector, rhs.myVector + myRows*myCols, myVector);
    }

    ptrMatrix& operator=(const ptrMatrix& rhs)
    {
        if (this == &rhs) return *this;
        ptrMatrix<T> temp(rhs);
        swap(temp);
        return *this;
    }

    ~ptrMatrix()
    {
        if (myVector) delete[] myVector;
        myVector = nullptr;
    }

    //  Move, move assign
    ptrMatrix(ptrMatrix&& rhs) : myRows(rhs.myRows), myCols(rhs.myCols), myVector(rhs.myVector)
    {
        rhs.myVector = nullptr;
    }

    ptrMatrix& operator=(ptrMatrix&& rhs)
    {
        if (this == &rhs) return *this;
        ptrMatrix<T> temp(move(rhs));
        swap(temp);
        return *this;
    }

    //  Swapper
    void swap(ptrMatrix& rhs)
    {
        swap(myVector, rhs.myVector);
        swap(myRows, rhs.myRows);
        swap(myCols, rhs.myCols);
    }

    //  Resizer
    void resize(const size_t rows, const size_t cols)
    {
        if (myRows * myCols < rows*cols)
        {
            ptrMatrix<T>(rows, cols).swap(*this);
        }
        else
        {
            myRows = rows;
            myCols = cols;
        }
    }

    //  Access
    size_t rows() const { return myRows; }
    size_t cols() const { return myCols; }
    //  So we can call matrix [i][j]
    T* operator[] (const size_t row) { return &myVector[row*myCols]; }
    const T* operator[] (const size_t row) const { return &myVector[row*myCols]; }

    //  Comparison
    bool operator==(const ptrMatrix& rhs) const
    {
        if (rows() != rhs.rows() || cols() != rhs.cols()) return false;
        for (size_t i = 0; i < rows(); ++i)
        {
            const double* ai = operator[](i);
            const double* bi = rhs[i];
            for (size_t j = 0; j < cols(); ++j)
            {
                if (!equals(ai[j], bi[j])) return false;
            }
        }
        return true;
    }

};

//  naive version of a matrix product, as seen in many libraries, including professional ones in investment banks
inline void matrixProductNaive(const ptrMatrix<double>& a, const ptrMatrix<double>& b, ptrMatrix<double>& c)
{
    const size_t rows = a.rows(), cols = b.cols(), n = a.cols();

    //  outermost loop on result rows
    for (size_t i = 0; i < rows; ++i)
    {
        const auto ai = a[i];
        auto ci = c[i];

        //  loop on result columns
        for (size_t j = 0; j < cols; ++j)
        {
            //  compute dot product
            double res = 0.0;
            for (size_t k = 0; k < n; ++k)
            {
                res += ai[k] * b[k][j];     //  note b[k][j] "jumps" through memory in the innermost loop - cache inneficiency
            }   //  dot
            //  set result
            c[i][j] = res;
        }   //  columns
    }   //  rows
}

//  reorder loops to avoid cache inneficiency - same algorithm and calculations otherwise
inline void matrixProductSmartNoVec(const ptrMatrix<double>& a, const ptrMatrix<double>& b, ptrMatrix<double>& c)
{
    const size_t rows = a.rows(), cols = b.cols(), n = a.cols();

    //  zero result first
    for (size_t i = 0; i < rows; ++i)
    {
        auto ci = c[i];
        for (size_t j = 0; j < cols; ++j)
        {
            ci[j] = 0;
        }
    }

    //  loop on result rows as before
    for (size_t i = 0; i < rows; ++i)
    {
        const auto ai = a[i];
        auto ci = c[i];

        //  then loop not on result columns but on dot product
        for (size_t k = 0; k < n; ++k)
        {
            const auto bk = b[k];
            const auto aik = ai[k]; //  we still jump when reading memory, but not in the innermost loop

            //  and finally over columns in innermost loop
#pragma loop(no_vector)     //  no vectorization to isolate impact of cache alone
            for (size_t j = 0; j < cols; ++j)
            {
                ci[j] += aik * bk[j];   //  no more jumping through memory
            }   //  columns
        }   //  dot
    }   //  rows
}

//  same but with vectorization
inline void matrixProductSmartVec(const ptrMatrix<double>& a, const ptrMatrix<double>& b, ptrMatrix<double>& c)
{
    const size_t rows = a.rows(), cols = b.cols(), n = a.cols();

    for (size_t i = 0; i < rows; ++i)
    {
        auto ci = c[i];
        for (size_t j = 0; j < cols; ++j)
        {
            ci[j] = 0;
        }
    }

    for (size_t i = 0; i < rows; ++i)
    {
        const auto ai = a[i];
        auto ci = c[i];

        for (size_t k = 0; k < n; ++k)
        {
            const auto bk = b[k];
            const auto aik = ai[k];

            //  the only difference is the absence of pragma: the compiler is free to vectorize
            for (size_t j = 0; j < cols; ++j)
            {
                ci[j] += aik * bk[j];
            }
        }
    }
}

//  same with multi-threading over outermost loop
inline void matrixProductSmartParallel(const ptrMatrix<double>& a, const ptrMatrix<double>& b, ptrMatrix<double>& c)
{
    const size_t rows = a.rows(), cols = b.cols(), n = a.cols();

    for (int i = 0; i < rows; ++i)
    {
        auto ci = c[i];
        for (size_t j = 0; j < cols; ++j)
        {
            ci[j] = 0;
        }
    }

    //  only difference is this open MP pragma, tells the compiler to send chunks of the loop across threads executing on different CPUs
    //  note the extreme simplicity of this technique
    //  however it is only relevant for simplistic multi-threading of simple loops
    //  to properly multi-thread something as complex as Monte-Carlo simulations, we need a more thorough solution than open MP
#pragma omp parallel for 
    for (int i = 0; i < rows; ++i)
    {
        const auto ai = a[i];
        auto ci = c[i];

        for (size_t k = 0; k < n; ++k)
        {
            const auto bk = b[k];
            const auto aik = ai[k];

            for (size_t j = 0; j < cols; ++j)
            {
                ci[j] += aik * bk[j];
            }
        }
    }
}

#include <iostream>
#include <stdlib.h>
#include <time.h>
using namespace std;

void matrixProductTester()
{
    const size_t na = 1000, ma = 1000, nb = 1000, mb = 1000;

    //  allocate
    ptrMatrix<double> a(na, ma), b(nb, mb), c1(na, mb), c2(na, mb), c3(na, mb), c4(na, mb);

    //  randomly fill
    srand(12345);
    for (size_t i = 0; i < na; ++i)
    {
        auto ai = a[i];

        for (size_t j = 0; j < ma; ++j)
        {
            ai[j] = double(rand()) / RAND_MAX;
        }
    }

    for (size_t i = 0; i < nb; ++i)
    {
        auto bi = b[i];

        for (size_t j = 0; j < mb; ++j)
        {
            bi[j] = double(rand()) / RAND_MAX;
        }
    }

    //  calculate and time
    {
        cout << "Naive calculation starting" << endl;
        time_t t1 = clock();
        matrixProductNaive(a, b, c1);
        time_t t2 = clock();
        cout << "Naive calculation complete, MS = " << t2 - t1 << endl;
    }

    {
        cout << "Smart calculation starting" << endl;
        time_t t1 = clock();
        matrixProductSmartNoVec(a, b, c2);
        time_t t2 = clock();
        cout << "Smart calculation complete, MS = " << t2 - t1 << endl;
    }

    {
        cout << "Vectorized calculation starting" << endl;
        time_t t1 = clock();
        matrixProductSmartVec(a, b, c3);
        time_t t2 = clock();
        cout << "Vectorized calculation complete, MS = " << t2 - t1 << endl;
    }

    {
        cout << "Parallel calculation starting" << endl;
        time_t t1 = clock();
        matrixProductSmartParallel(a, b, c4);
        time_t t2 = clock();
        cout << "Parallel calculation complete, MS = " << t2 - t1 << endl;
    }

    //  check
    cout << "Check = " << (c1 == c2) << " , " << (c2 == c3) << " , " << (c3 == c4) << endl;
    cout << "Check2 = " << c1[99][98] << " , " << c2[99][98] << " , " << c3[99][98] << " , " << c4[99][98] << endl;

}