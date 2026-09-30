#include "pseudo_products.h"

#include "io_data.h"
#include "roots_and_norms_calculations.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>

template <typename Ttype>
void read_pseudo_products_matrix(
    const std::filesystem::path &file_path,
    std::vector<std::vector<std::vector<Ttype>>>& pseudo_products_matrix)
{
    static_assert(std::is_trivially_copyable<Ttype>::value,
                  "read_pseudo_products_matrix requires trivially copyable scalar types");

    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("Could not open file for reading: " + file_path.string());
    }

    char magic[8];
    std::uint8_t version = 0;
    std::uint8_t scalar_size = 0;
    std::uint32_t n_size = 0;

    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&version), sizeof(version));
    in.read(reinterpret_cast<char *>(&scalar_size), sizeof(scalar_size));
    in.read(reinterpret_cast<char *>(&n_size), sizeof(n_size));

    if (!in.good())
    {
        throw std::runtime_error("Invalid pseudo-products matrix header: " + file_path.string());
    }

    constexpr char expected_magic[8] = {'P', 'P', 'M', 'A', 'T', 'R', 'X', '1'};
    if (std::memcmp(magic, expected_magic, sizeof(expected_magic)) != 0)
    {
        throw std::runtime_error("Invalid pseudo-products matrix magic in file: " + file_path.string());
    }

    if (version != 1)
    {
        throw std::runtime_error("Unsupported pseudo-products matrix format version: " + std::to_string(version));
    }

    const bool exact_match = (scalar_size == sizeof(Ttype));
    const bool double_widening = (!exact_match && scalar_size == sizeof(double));
    if (!exact_match && !double_widening)
    {
        throw std::runtime_error("Pseudo-products matrix scalar size mismatch in file: " + file_path.string());
    }

    std::vector<std::uint32_t> m_sizes(n_size, 0);
    if (n_size > 0)
    {
        in.read(reinterpret_cast<char *>(m_sizes.data()), static_cast<std::streamsize>(n_size * sizeof(std::uint32_t)));
        if (!in.good())
        {
            throw std::runtime_error("Invalid pseudo-products matrix dimensions block: " + file_path.string());
        }
    }

    pseudo_products_matrix.resize(n_size);

    for (std::uint32_t n = 0; n < n_size; n++)
    {
        const std::uint32_t m_size = m_sizes[n];
        pseudo_products_matrix[n].resize(m_size);
        for (std::uint32_t m1 = 0; m1 < m_size; m1++)
        {
            auto &row = pseudo_products_matrix[n][m1];
            row.resize(m_size);
            if (exact_match)
            {
                const std::streamsize row_num_bytes = static_cast<std::streamsize>(m_size * sizeof(Ttype));
                in.read(reinterpret_cast<char *>(row.data()), row_num_bytes);
            }
            else
            {
                // File was written as double; widen each value to Ttype (e.g. long double).
                std::vector<double> tmp(m_size);
                in.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(m_size * sizeof(double)));
                for (std::uint32_t m2 = 0; m2 < m_size; m2++)
                    row[m2] = static_cast<Ttype>(tmp[m2]);
            }
            if (!in.good())
            {
                throw std::runtime_error("Unexpected EOF while reading pseudo-products matrix payload: " + file_path.string());
            }
        }
    }

    // return pseudo_products_matrix;
}

template <typename Ttype>
void compute_and_save_pseudo_products_matrix(
    const std::filesystem::path &roots_path,
    const std::filesystem::path &gauss_data_path,
    const std::filesystem::path &output_path)
{
    std::vector<std::vector<std::vector<Ttype>>> pseudo_products_matrix = compute_pseudo_products_matrix<Ttype>(roots_path, gauss_data_path);
    save_pseudo_products_matrix(output_path, pseudo_products_matrix);
}

template <typename Ttype>
void save_pseudo_products_matrix(
    const std::filesystem::path &output_path,
    const std::vector<std::vector<std::vector<Ttype>>> &pseudo_products_matrix)
{
    static_assert(std::is_trivially_copyable<Ttype>::value,
                  "save_pseudo_products_matrix requires trivially copyable scalar types");

    const std::filesystem::path parent_dir = output_path.parent_path();
    if (!parent_dir.empty())
    {
        std::error_code ec;
        std::filesystem::create_directories(parent_dir, ec);
        if (ec)
        {
            throw std::runtime_error("Could not create directory: " + parent_dir.string());
        }
    }

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
    {
        throw std::runtime_error("Could not open file for writing: " + output_path.string());
    }

    // Binary layout (little overhead, fast runtime loading):
    // [8B magic][1B version][1B scalar_size][4B n_size][n_size * 4B m_size]
    // [raw values in order n -> m1 -> m2]
    constexpr char magic[8] = {'P', 'P', 'M', 'A', 'T', 'R', 'X', '1'};
    constexpr std::uint8_t version = 1;
    const std::uint8_t scalar_size = static_cast<std::uint8_t>(sizeof(Ttype));

    if (pseudo_products_matrix.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::runtime_error("Pseudo-products matrix n dimension is too large to serialize");
    }

    const std::uint32_t n_size = static_cast<std::uint32_t>(pseudo_products_matrix.size());

    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&version), sizeof(version));
    out.write(reinterpret_cast<const char *>(&scalar_size), sizeof(scalar_size));
    out.write(reinterpret_cast<const char *>(&n_size), sizeof(n_size));

    std::vector<std::uint32_t> m_sizes;
    m_sizes.reserve(n_size);
    for (std::uint32_t n = 0; n < n_size; n++)
    {
        const auto m_size_raw = pseudo_products_matrix[n].size();
        if (m_size_raw > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        {
            throw std::runtime_error("Pseudo-products matrix m dimension is too large to serialize");
        }

        const std::uint32_t m_size = static_cast<std::uint32_t>(m_size_raw);
        m_sizes.push_back(m_size);
        out.write(reinterpret_cast<const char *>(&m_size), sizeof(m_size));
    }

    for (std::uint32_t n = 0; n < n_size; n++)
    {
        const std::uint32_t m_size = m_sizes[n];
        for (std::uint32_t m1 = 0; m1 < m_size; m1++)
        {
            const auto &row = pseudo_products_matrix[n][m1];
            if (row.size() != m_size)
            {
                throw std::runtime_error("Pseudo-products matrix is not square in m1/m2 for n = " + std::to_string(n));
            }

            const auto *row_bytes = reinterpret_cast<const char *>(row.data());
            const std::streamsize row_num_bytes = static_cast<std::streamsize>(row.size() * sizeof(Ttype));
            out.write(row_bytes, row_num_bytes);
        }
    }

    if (!out.good())
    {
        throw std::runtime_error("Error while writing pseudo-products matrix: " + output_path.string());
    }
}

template <typename Ttype>
std::vector<std::vector<std::vector<Ttype>>> compute_pseudo_products_matrix(
    const std::filesystem::path &roots_path,
    const std::filesystem::path &gauss_data_path)
{
    // Load the roots and gaussian data
    std::vector<std::vector<Ttype>> roots;
    std::vector<std::vector<Ttype>> gaussian_data;

    readValuesFromFile(roots_path, roots);
    readValuesFromFile(gauss_data_path, gaussian_data);

    std::vector<Ttype> gauss_weights;
    std::vector<Ttype> gauss_points;
    gauss_weights.reserve(gaussian_data.size());
    gauss_points.reserve(gaussian_data.size());
    for (const auto &row : gaussian_data)
    {
        if (row.size() < 2)
        {
            throw std::runtime_error("Invalid gaussian data row in file: expected at least 2 columns");
        }
        gauss_weights.push_back(row[0]);
        gauss_points.push_back(row[1]);
    }

    // Compute the pseudo_products
    unsigned nMax = roots.size();
    std::vector<std::vector<std::vector<Ttype>>> pseudo_products_matrix;
    pseudo_products_matrix.resize(nMax);
    for (unsigned n = 0; n < nMax; n++)
    {
        unsigned mMax = roots[n].size();
        pseudo_products_matrix[n].assign(mMax, std::vector<Ttype>(mMax, static_cast<Ttype>(0)));
        for (unsigned m1 = 0; m1 < mMax; m1++)
        {
            // Matrix is symmetric in (m1, m2): compute only the upper triangle.
            for (unsigned m2 = m1; m2 < mMax; m2++)
            {
                const Ttype val =
                    get_integration_product_r_direct(n, roots[n][m1], roots[n][m2], gauss_weights, gauss_points);
                pseudo_products_matrix[n][m1][m2] = val;
                pseudo_products_matrix[n][m2][m1] = val;
            }
            if ((m1 + 1) % 20 == 0 || m1 + 1 == mMax)
            {
                std::cout << "  n=" << n << ": row " << (m1 + 1) << "/" << mMax << " done." << std::endl;
            }
        }
        std::cout << "n = " << n << " done." << std::endl;
    }

    return pseudo_products_matrix;
}

// Templates instantiation
template void read_pseudo_products_matrix<double>(
    const std::filesystem::path &,
    std::vector<std::vector<std::vector<double>>>&);

template void read_pseudo_products_matrix<long double>(
    const std::filesystem::path &,
    std::vector<std::vector<std::vector<long double>>>&);

template void compute_and_save_pseudo_products_matrix<double>(
    const std::filesystem::path &,
    const std::filesystem::path &,
    const std::filesystem::path &);

template void compute_and_save_pseudo_products_matrix<long double>(
    const std::filesystem::path &,
    const std::filesystem::path &,
    const std::filesystem::path &);

template void save_pseudo_products_matrix<double>(
    const std::filesystem::path &,
    const std::vector<std::vector<std::vector<double>>> &pseudo_products_matrix);

template void save_pseudo_products_matrix<long double>(
    const std::filesystem::path &,
    const std::vector<std::vector<std::vector<long double>>> &pseudo_products_matrix);

template std::vector<std::vector<std::vector<double>>> compute_pseudo_products_matrix<double>(
    const std::filesystem::path &,
    const std::filesystem::path &);

template std::vector<std::vector<std::vector<long double>>> compute_pseudo_products_matrix<long double>(
    const std::filesystem::path &,
    const std::filesystem::path &);
