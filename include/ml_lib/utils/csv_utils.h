#ifndef ML_LIB_CSV_UTILS_H
#define ML_LIB_CSV_UTILS_H

#include <csv.hpp>
#include <vector>
#include <string>
#include <functional>

namespace ml_lib {
namespace utils {

/**
 * @brief Utility functions for reading CSV files for ML applications
 */
class CSVUtils {
public:
    /**
     * @brief Read CSV and parse all columns as numeric data
     *
     * @param filename Path to CSV file
     * @param has_header Whether the first row is a header
     * @return std::vector<std::vector<double>> Numeric data
     */
    static std::vector<std::vector<double>> readNumeric(
        const std::string& filename,
        bool has_header = true);

    /**
     * @brief Read CSV with custom column parsers
     *
     * @param filename Path to CSV file
     * @param column_parsers Vector of parsing functions for each column
     * @param has_header Whether the first row is a header
     * @return std::vector<std::vector<double>> Parsed data
     */
    static std::vector<std::vector<double>> readWithParsers(
        const std::string& filename,
        const std::vector<std::function<double(const std::string&)>>& column_parsers,
        bool has_header = true);

    /**
     * @brief Read CSV selecting specific columns
     *
     * @param filename Path to CSV file
     * @param column_indices Indices of columns to read (0-based)
     * @param has_header Whether the first row is a header
     * @return std::vector<std::vector<double>> Selected columns
     */
    static std::vector<std::vector<double>> readColumns(
        const std::string& filename,
        const std::vector<size_t>& column_indices,
        bool has_header = true);

    /**
     * @brief Read CSV and split into features and target
     *
     * @param filename Path to CSV file
     * @param target_column Index of target column
     * @param features Output: feature data
     * @param target Output: target data
     * @param has_header Whether the first row is a header
     * @return true if successful
     */
    static bool readFeatureTarget(
        const std::string& filename,
        size_t target_column,
        std::vector<std::vector<double>>& features,
        std::vector<double>& target,
        bool has_header = true);

    // Common parsing utilities
    static double parseBinary(const std::string& value);
    static double parseYesNo(const std::string& value);
    static double parseTrueFalse(const std::string& value);
};

} // namespace utils
} // namespace ml_lib

#endif // ML_LIB_CSV_UTILS_H
