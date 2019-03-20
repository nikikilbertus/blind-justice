#pragma once
#include <boost/numeric/ublas/matrix.hpp>
#include <random>

// fills matrix with pseudorandomly generated elements
template<class Matrix, class Generator>
void randomize_matrix(Generator&& r, Matrix& m) {
  std::uniform_int_distribution<typename Matrix::value_type> dist;
  for(size_t i = 0; i < m.size1(); i++) {
    for(size_t j = 0; j < m.size2(); j++) {
      m(i, j) = dist(r);
    }
  }
}
