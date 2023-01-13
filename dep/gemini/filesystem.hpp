/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef FILESYSTEM_HPP
#define FILESYSTEM_HPP

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

inline bool file_exists(std::string filename) {
  struct stat st;
  if (stat(filename.c_str(), &st) != 0) {
    printf("%s not exist!\n", filename.c_str());
  }
  return stat(filename.c_str(), &st) == 0;
}

inline long file_size(std::string filename) {
  struct stat st;
  if (stat(filename.c_str(), &st) != 0) {
    printf("%s file_size is not correct!\n", filename.c_str());
  }
  assert(stat(filename.c_str(), &st) == 0);
  return st.st_size;
}

#endif