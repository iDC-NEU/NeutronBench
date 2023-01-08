#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <random>
#include <vector>

#include "type.hpp"

template <typename T>
void shuffle_vec(std::vector<T>& vec) {
  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  unsigned seed = 2000;
  static thread_local std::mt19937 generator(seed);
  std::shuffle(vec.begin(), vec.end(), generator);
}

#endif