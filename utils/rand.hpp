#ifndef _RAND_HPP_
#define _RAND_HPP_

#include <random>

int random_uniform_int(const int min = 0, const int max = 1) {
  // thread_local std::default_random_engine generator;
  unsigned seed = 2000;
  static thread_local std::mt19937 generator(seed);
  // static thread_local std::mt19937 generator;
  std::uniform_int_distribution<int> distribution(min, max);
  return distribution(generator);
}

// random from [lower, upper)
template <typename T>
T rand_int_seed(T lower, T upper) {
  assert(lower < upper);
  static thread_local std::mt19937 generator(2000);
  std::uniform_int_distribution<T> distribution(lower, upper - 1);
  return distribution(generator);
}

template <typename T>
T rand_int_seed(T upper) {
  return rand_int_seed<T>(0, upper);
}

template <typename T>
T rand_int(T lower, T upper) {
  assert(lower < upper);
  // static thread_local unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static thread_local std::mt19937 generator;
  std::uniform_int_distribution<T> distribution(lower, upper - 1);
  return distribution(generator);
}

template <typename T>
T rand_int(T upper) {
  return rand_int<T>(0, upper);
}

#endif