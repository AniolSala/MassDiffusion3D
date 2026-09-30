#include "../src/io_data.h"

template <typename Ttype>
void writeValuesToFile(const std::string& filename, const std::vector<std::vector<Ttype>>& data) {
    std::ofstream file(filename);

    if (!file) {
        // std::cerr << "Error: Could not open the file \"" << filename << "\" for writing.\n";
        throw std::runtime_error("Could not open file " + filename + " for writing.");
        return;
    }

    file << std::scientific << std::setprecision(std::numeric_limits<Ttype>::max_digits10);
    for (const auto& row : data) {
        for (const auto& value : row) {
            file << value << " ";
        }
        file << "\n";
    }

    file.close();
    // std::cout << "Data successfully written to \"" << filename << "\".\n";
}

template <typename Ttype>
void writeValuesToFile(const std::string& filename, const std::vector<Ttype>& data) {
    std::ofstream file(filename);

    if (!file) {
        std::cerr << "Could not open the file \"" << filename << "\" for writing.\n";
        return;
    }

    for (const auto& value : data) {
        file << std::setprecision(30) << value << " "; // Write each value with high precision
        file << "\n"; // Add a newline after each row
    }

    file.close();
    // std::cout << "Data successfully written to \"" << filename << "\".\n";
}

template <typename Ttype>
void readValuesFromFile(const std::string& filename, std::vector<std::vector<Ttype>>& data) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    // Ensure input streams handle full precision
    inputFile.precision(std::numeric_limits<Ttype>::max_digits10);

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream lineStream(line);
        lineStream.precision(std::numeric_limits<Ttype>::max_digits10); // For line stream

        std::vector<Ttype> row;
        Ttype value;
        while (lineStream >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    inputFile.close();
}

template <typename Ttype>
void readValuesFromFile(const std::string& filename, std::vector<Ttype>& data) {
    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    // Ensure input streams handle full precision
    inputFile.precision(std::numeric_limits<Ttype>::max_digits10);

    Ttype value;
    while (inputFile>> value) {
        data.push_back(value);
    }

    inputFile.close();
}

template void writeValuesToFile<double>(const std::string& filename, const std::vector<double>& data);
template void writeValuesToFile<double>(const std::string& filename, const std::vector<std::vector<double>>& data);
template void readValuesFromFile<double>(const std::string& filename, std::vector<double>& data);
template void readValuesFromFile<double>(const std::string& filename, std::vector<std::vector<double>>& data);

template void writeValuesToFile<long double>(const std::string& filename, const std::vector<long double>& data);
template void writeValuesToFile<long double>(const std::string& filename, const std::vector<std::vector<long double>>& data);
template void readValuesFromFile<long double>(const std::string& filename, std::vector<long double>& data);
template void readValuesFromFile<long double>(const std::string& filename, std::vector<std::vector<long double>>& data);
