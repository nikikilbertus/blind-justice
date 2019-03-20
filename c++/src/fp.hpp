#pragma once
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using namespace boost::numeric::ublas;

template <typename T>
void fp2double(T x_fp, double &x, size_t &p) {
    x = double((int32_t) x_fp) / (1ll << p);
    //cout << "fp2double(" << x_fp << "," << p << ") = " << x << endl;
}

template <typename T>
void double2fp(double x, T &x_fp, size_t &p) {
    x_fp = (x * (1ll << p));
    /*cout << "double2fp(" << x << "," << p << ") = " << x_fp << endl;
    // test:
    double x_test = 99.99;
    fp2double(x_fp, x_test, p);*/
}


// This assumes memory for M_fp has been allocated
template <typename T>
void double_matrix2fp(matrix<double> &M, matrix<T> &M_fp, size_t precision) {
    for (size_t i = 0; i < M.size1(); i++) {
        for (size_t j = 0; j < M.size2(); j++) {
        double2fp(M(i, j), M_fp(i, j), precision);
        }
    }
}

// This assumes memory for M_fp has been allocated
template <typename T>
void fp_matrix2double(matrix<T> &M_fp, matrix<double> &M, size_t precision) {
    for (size_t i = 0; i < M.size1(); i++) {
        for (size_t j = 0; j < M.size2(); j++) {
        fp2double(M_fp(i, j), M(i, j), precision);
        }
    }
}
