#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <random>
#include <vector>

#include "type.hpp"

template <typename T>
void shuffle_vec(std::vector<T>& vec) {
  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static thread_local std::mt19937 generator;
  std::shuffle(vec.begin(), vec.end(), generator);
}

template <typename T>
void shuffle_vec_seed(std::vector<T>& vec) {
  static thread_local std::mt19937 generator(2000);
  std::shuffle(vec.begin(), vec.end(), generator);
}

#endif