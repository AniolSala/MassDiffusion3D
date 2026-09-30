#ifndef READ_VALUES_H
#define READ_VALUES_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <limits>



template <typename Ttype>
void writeValuesToFile(const std::string& filename, const std::vector<std::vector<Ttype>>& data);
template <typename Ttype>
void writeValuesToFile(const std::string& filename, const std::vector<Ttype>& data);
template <typename Ttype>
void readValuesFromFile(const std::string& filename, std::vector<std::vector<Ttype>>& data);
template <typename Ttype>
void readValuesFromFile(const std::string& filename, std::vector<Ttype>& data);

#endif