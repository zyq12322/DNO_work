#ifndef VAR_OP_FUNCTION_H
#define VAR_OP_FUNCTION_H
#include <string>
#include <cctype>
#include <algorithm>
#include <vector>
#include <variant>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <charconv>
#include <stdexcept>
#include <iostream>
#include <type_traits>
#include <regex>
#include <fftw3.h>
#include <complex>
#include <utility>
#include <memory>

const std::complex<double> ImUnit(0.0, 1.0);

using IDBSVariant = std::variant<int,double,bool,std::string>;

using IDBSVector = std::vector<IDBSVariant>;
// define inline functions
// removes leading and trailing whitespace characters from a string.
/**
 * @brief Removes leading and trailing whitespace characters from a string.
 *
 * This function scans the input string from the beginning to find the first
 * character that is not a whitespace (space, tab, newline, carriage return,
 * form feed, vertical tab). It then scans from the end to find the last
 * non-whitespace character. The substring between these two positions is
 * returned. If the string contains only whitespace, an empty string is returned.
 *
 * @param s The input string to be trimmed.
 * @return A new string with leading and trailing whitespace removed.
 *         Returns an empty string if the input is empty or consists solely
 *         of whitespace characters.
 *
 * @note The function does not modify the original string.
 * @note The set of whitespace characters includes: space (' '), tab ('\t'),
 *       newline ('\n'), carriage return ('\r'), form feed ('\f'), vertical tab ('\v').
 */
inline std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) return "";
    auto end = s.find_last_not_of(" \t\n\r\f\v");
    return s.substr(start, end - start + 1);
}
/**
 * @brief Converts a string to lowercase.
 *
 * This function returns a copy of the input string where all characters
 * are converted to lowercase using std::tolower.
 *
 * @param s Input string
 * @return Lowercase version of the input string
 */
inline std::string to_lower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                [](unsigned char c) { return std::tolower(c); });
    return result;
}
/**
 * @brief Checks whether a string represents a valid integer.
 *
 * This function performs a syntactic validation to determine whether the input
 * string is a valid base-10 integer representation. It allows:
 * - Optional leading '+' or '-' sign
 * - One or more digits (0-9)
 *
 * The function does NOT allow:
 * - Empty strings
 * - Sign-only strings (e.g., "+" or "-")
 * - Any non-digit characters
 *
 * Examples of valid inputs:
 * - "0"
 * - "123"
 * - "-42"
 * - "+99"
 *
 * Examples of invalid inputs:
 * - ""
 * - "+"
 * - "-"

* - "12a3"
* - "3.14"
*
* @param s Input string to validate
* @return true if the string is a valid integer, false otherwise
*
* @note This is a syntactic check only and does not verify numeric range
*       (e.g., overflow beyond int/long limits is not considered).
*/
inline bool is_integer(const std::string& s) {
    // Reject empty strings (nothing to parse)
    if (s.empty()) return false;
    // start from the first character; allow an optional sign prefix.
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i++;
    // only contains a "+" or "-", invalid-return false;
    if (i >= s.size()) return false;

    // check that every remaining character is a digit.
    for (; i < s.size(); ++i) {
        // cast unsigned char to ensure safe argument for std::isdigit
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}
/**
 * @brief Checks whether a string represents a valid floating-point number.
 *
 * This function performs a manual lexical validation of a floating-point number
 * in decimal or scientific notation. It supports:
 * - Optional leading '+' or '-' sign
 * - Optional decimal point ('.')
 * - Optional exponent part using 'e' or 'E', with optional sign
 *
 * Valid examples:
 * - "123"
 * - "-123.45"
 * - "+0.99"
 * - "1e10"
 * - "-3.14E-2"
 *
 * Invalid examples:
 * - "."
 * - "e10"
 * - "1e"
 * - "abc"
 *
 * @param s Input string to validate
 * @return true if the string is a valid floating-point number, false otherwise
 *
 * @note This function does not support special values like "inf", "infinity", or "NaN".
 * @note This is a syntactic check only; it does not guarantee safe conversion precision.
 */
inline bool is_float(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i++;
    if (i >= s.size()) return false;
    bool has_dot = false;
    bool has_exp = false;
    bool has_digit_before_exp = false;
    bool has_digit_after_exp = false;

    for (; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (std::isdigit(c)) {
            if (!has_exp)
                has_digit_before_exp = true;
            else
                has_digit_after_exp = true;
            continue;
        }
        if (c == '.') {
            if (has_dot || has_exp) return false;
            has_dot = true;
            continue;
        }
        if (c == 'e' || c == 'E') {
            if (has_exp || !has_digit_before_exp) return false;
            has_exp = true;
            if (i + 1 < s.size() && (s[i+1] == '+' || s[i+1] == '-')) {
                ++i;
            }
            continue;
        }
        return false;
    }
    if (has_exp && !has_digit_after_exp) return false;
    if (!has_digit_before_exp) return false;
    return true;
}
/** 
This function uses std::visit to handle each alternative type in the variant.
    -bool values are printed as "true" or "false" instead of 1/0
    -int, double, and std::string are printed using the stream's default formatting.
    @param os Output stream where the value will be written.
    @param v Variant holding either int, double, bool or std::string.
*/
inline void print_variantIDBS_item(std::ostream& os, const IDBSVariant& v) {
    std::visit([&os](auto&& arg) {
    if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, bool>)
        os << (arg ? "true" : "false");
    else
        os << arg;
    }, v);
}
/** 
@brief Prints a vector of std::variant<int, double, bool, std::string> to an output stream.
Elements are printed using print_variantIDBS_item() and separated by a configurable delimiter.
@param os Output stream where the formatted data will be written
@param vec_variant Vector of variants to print.
@param delimiter String used to separate elements(default: ",").
*/
inline void print_variantIDBS_vector(
    std::ostream& os, 
    const IDBSVector& vec_variant,
    const std::string& delimiter = ",")
{
    bool first = true;
    for (const auto& item : vec_variant) {
        if (!first)
            os << delimiter;
        first = false;

        print_variantIDBS_item(os, item);
    }
}

/**
 * @brief Converts a std::variant<int, double, bool, std::string> to a double value.
 *
 * This function extracts a numeric representation from the variant:
 * - int values are converted to double using static_cast
 * - double values are returned directly
 * - bool values are converted to 1.0 (true) or 0.0 (false)
 *
 * If the variant holds a std::string or any other non-numeric type,
 * the function throws a std::runtime_error.
 *
 * @param v Variant containing int, double, bool, or std::string
 * @return The numeric value represented as a double
 * @throws std::runtime_error if the variant does not contain a numeric type
 */
inline double variantIDBS_to_double(
    const IDBSVariant& v)
{
    if (auto p = std::get_if<int>(&v)) {
        return static_cast<double>(*p);
    } 
    else if (auto p = std::get_if<double>(&v)) {
        return *p;
    }
    else if (auto p = std::get_if<bool>(&v)) {
        return *p ? 1.0 : 0.0;
    }
    throw std::runtime_error("variant does not contain a numeric type");
    
}
/**
 * @brief Converts a std::variant<int, double, bool, std::string> to a string.
 *
 * This function converts the value stored in the variant into its string representation:
 * - bool values are converted using "true"/"false" (std::boolalpha)
 * - double values are formatted with high precision (15 digits)
 * - int and std::string values are streamed directly
 *
 * The function uses std::ostringstream and std::visit to safely handle all supported types.
 *
 * @param var Variant containing int, double, bool, or std::string
 * @return String representation of the stored value
 *
 * @note Floating-point formatting may use scientific notation depending on the value.
 */
inline std::string variantIDBS_to_string(const IDBSVariant& var) {
    std::ostringstream oss;
    std::visit([&oss](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, bool>) {
            oss << std::boolalpha << arg;
        } else if constexpr (std::is_same_v<T, double>) {
            oss << std::setprecision(17) << arg;
        } else {
            oss << arg;
        }
    }, var);
    return oss.str();
}

/**
 * @brief Splits a string by commas and trims each token.
 *
 * The function divides the input string at each ',' delimiter,
 * trims whitespace from each token, and stores non-empty results.
 *
 * @param s Input string containing comma-separated values
 * @param parts Output vector that will be filled with tokens
 */
inline void split_by_comma(const std::string& s,
                            std::vector<std::string>& parts)
{
    parts.clear();

    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty())
            parts.push_back(std::move(item));
    }
}

/**
 * @brief Extracts the substring before a given delimiter character.
 *
 * If the delimiter is found, returns the trimmed substring before it.
 * Otherwise, returns the entire trimmed string.
 *
 * @param s Input string
 * @param delim Delimiter character
 * @return Trimmed substring before the delimiter
 */
inline std::string get_key_before_char(const std::string& s, char delim)
{
    // find the first delim in s
    std::size_t pos = s.find(delim);

    if (pos == std::string::npos)
        return trim(s);

    return trim(s.substr(0, pos));
}

/**
 * @brief Extracts the substring after a given delimiter character.
 *
 * If the delimiter is found, returns the trimmed substring after it.
 * Otherwise, returns an empty string.
 *
 * @param s Input string
 * @param delim Delimiter character
 * @return Trimmed substring after the delimiter
 */
inline std::string get_val_after_char(const std::string& s, char delim)
{
    // find the first delim in s
    std::size_t pos = s.find(delim);

    if (pos == std::string::npos)
        return "";

    return trim(s.substr(pos + 1));
}

/**
 * @brief Safely attempts to parse an integer from a string.
 *
 * This function uses `std::from_chars` to perform fast, locale‑independent
 * conversion of the entire string `s` to an `int` value. No leading or trailing
 * whitespace is allowed; the string must consist solely of an optional sign
 * followed by one or more decimal digits.
 *
 * Detailed behavior:
 * - An empty string returns `false`.
 * - A string containing only a sign (`"+"` or `"-"`) returns `false` (no digits).
 * - Leading whitespace is **not** skipped (returns `false`).
 * - Trailing characters after the integer (e.g., `"123abc"`) return `false`.
 * - The integer may be preceded by an optional `'+'` or `'-'` sign.
 * - If the parsed value is outside the representable range of `int`,
 *   `std::from_chars` reports `std::errc::result_out_of_range` and the function
 *   returns `false` (no undefined behavior).
 * - On success, `out` is written with the parsed value and `true` is returned.
 *
 * @param s   Input string to parse. Must contain a valid integer representation
 *            with no extra characters (including whitespace).
 * @param out Reference to an `int` that will receive the parsed value on success.
 *            Its content is unchanged if parsing fails.
 *
 * @return `true` if the entire string was successfully parsed as an `int`;
 *         `false` otherwise.
 *
 * @note This function is locale‑insensitive and does not depend on the global
 *       C locale. It does not skip whitespace – use `trim()` beforehand if needed.
 * @note Overflow and underflow are detected and cause a failure (returns `false`).
 *
 * @see std::from_chars
 */
inline bool try_parse_int(const std::string& s, int& out)
{
    const char* start = s.data();
    const char* end = start + s.size();

    if (*start == '+') ++start;
    if (start == end) return false;
    auto [p, ec] = std::from_chars(start, end, out);

    return ec == std::errc() && p == end;
}

/**
 * @brief Safely attempts to parse a double from a string.
 *
 * Uses std::from_chars (if supported) for fast, locale-independent parsing.
 * Requires the entire string to represent a valid floating-point number.
 *
 * @param s Input string
 * @param out Output double value if parsing succeeds
 * @return true if parsing succeeds, false otherwise
 */
inline bool try_parse_double(const std::string& s, double& out)
{
    const char* start = s.data();
    const char* end = start + s.size();

    if (*start == '+') ++start;
    if (start == end) return false;
    auto [p, ec] = std::from_chars(start, end, out);

    return ec == std::errc() && p == end;
}

/**
 * @brief Parses a string into a std::variant<int, double, bool, std::string>.
 *
 * Parsing order:
 * 1. "true"/"false" (case-insensitive)
 * 2. Integer
 * 3. Double
 * 4. Fallback to string
 *
 * @param s Input string
 * @return Parsed variant value
 */
inline IDBSVariant
parse_string_toIDBS(const std::string& s)
{
    std::string trimmed_s = trim(s);
    std::string lower = to_lower(trimmed_s);

    if (lower == "true")  return true;
    if (lower == "false") return false;

    int ival;
    if (try_parse_int(trimmed_s, ival))
        return ival;

    double dval;
    if (try_parse_double(trimmed_s, dval))
        return dval;

    return trimmed_s;
}

/**
 * @brief Divides a numeric value inside a variant vector by a divisor.
 *
 * The function accesses the element at the specified index and:
 * - If it is an int, converts it to double and divides
 * - If it is a double, divides directly
 *
 * @param vec Vector of variants
 * @param divisor Value used to divide the element
 * @param index Index of the element to modify
 *
 * @throws std::out_of_range if index is invalid
 * @throws std::runtime_error if the value is not numeric
 */
inline void divide_at_indexofIDBS(
    IDBSVector& vec,
    double divisor,
    size_t index)
{
    if (index >= vec.size())
        throw std::out_of_range("index out of range");

    if (divisor == 0.0)
        throw std::runtime_error("division by zero");

    auto& v = vec[index];

    if (auto* p = std::get_if<int>(&v)) {
        v = static_cast<double>(*p) / divisor;
    }
    else if (auto* p = std::get_if<double>(&v)) {
        v = *p / divisor;
    }
    else {
        throw std::runtime_error("variant does not contain a numeric type");
    }
}

namespace fs = std::filesystem;
/**
 * @brief Ensures a directory exists and is safe for writing simulation results.
 *
 * This function performs the following operations:
 *   - Resolves the input path to an absolute form.
 *   - If the directory already exists:
 *       - Verifies it is a directory (not a file or symlink).
 *       - Scans the directory for any regular file whose name matches the
 *         protected pattern `^[23]d_(eta|phis)\.txt$`. If such a file is found,
 *         an exception is thrown to prevent accidental overwriting of previous
 *         simulation outputs.
 *       - If the directory is not empty but contains no protected files,
 *         a warning is printed to `std::cerr`.
 *   - If the directory does not exist, it is created recursively (including any
 *     missing parent directories).
 *
 * The function is intended to be called before writing result files (e.g., time
 * series, spectral data) into the specified output folder.
 *
 * @param dir_path The directory path to check or create.
 *
 * @throws std::runtime_error if:
 *   - The path cannot be resolved to an absolute path.
 *   - A filesystem error occurs during existence or directory attribute checks.
 *   - The path exists but is not a directory.
 *   - The directory contains at least one file matching the protected pattern.
 *   - The directory cannot be created (e.g., permission denied, invalid path).
 *
 * @note The protected pattern is `^[23]d_(eta|phis)\.txt$`. It matches files like
 *       `2d_eta.txt`, `3d_phis.txt`, etc. These are considered critical result
 *       files that must not be overwritten.
 *
 * @warning The function uses a static `std::regex` object for pattern matching.
 *          The regex is compiled only once per program execution.
 */
inline void check_or_create_output_directory(const std::string& dir_path)
{
    static const std::regex protected_pattern(
        R"(^[23]d_(eta|phis)\.txt$)"
    );

    std::error_code ec;

    fs::path p = fs::absolute(dir_path, ec);

    if (ec)
    {
        throw std::runtime_error(
            "Failed to resolve directory path: " + dir_path
        );
    }

    // Directory already exists
    if (fs::exists(p, ec))
    {
        if (ec)
        {
            throw std::runtime_error(
                "Filesystem error while checking existence: "
                + ec.message()
            );
        }

        if (!fs::is_directory(p, ec))
        {
            throw std::runtime_error(
                "Path exists but is not a directory: "
                + p.string()
            );
        }

        bool contains_protected_file = false;

        auto it = fs::directory_iterator(p, ec);
        if (ec)
            throw std::runtime_error("Failed to open directory: " + ec.message());

        for (const auto& entry : it)
        {
            if (ec)
            {
                throw std::runtime_error(
                    "Failed to iterate directory: "
                    + ec.message()
                );
            }

            if (!entry.is_regular_file(ec))
                continue;

            const auto filename =
                entry.path().filename().string();

            if (std::regex_match(filename, protected_pattern))
            {
                contains_protected_file = true;
                break;
            }

        }

        if (contains_protected_file)
        {
            throw std::runtime_error(
                "Directory already contains protected result files: "
                + p.string()
            );
        }
        ec.clear();

        bool empty = fs::is_empty(p, ec);

        if (ec)
        {
            throw std::runtime_error(
                "Failed to determine if directory is empty: "
                + ec.message()
            );
        }

        if (!empty)
        {
            std::cerr
                << "Warning: directory is not empty:\n"
                << "  " << p << '\n'
                << "Result files will be written into it.\n";
        }

        return;
    }

    // Create directory tree
    if (!fs::create_directories(p, ec))
    {
        if (ec)
        {
            throw std::runtime_error(
                "Failed to create directory: "
                + p.string()
                + " (" + ec.message() + ")"
            );
        }
    }
}
/**
 * @brief Element-wise vector multiplication kernel.
 *
 * Computes the Hadamard (element-wise) product of vectors @p a and @p b
 * and stores the result in @p c:
 *
 *     c = a ⊙ b
 *
 * where
 *
 *     c[i] = a[i] * b[i].
 *
 * @param a Input vector.
 * @param b Input vector.
 * @param c Output vector.
 *
 * @pre a.size() == b.size() == c.size()
 *
 * @warning No size checking is performed for performance reasons.
 */
inline void mul_loop(const std::vector<double>& a,
    const std::vector<double>& b,
    std::vector<double>& c) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Two vectors must have same size in mul_loop!");
    }
    size_t n = a.size();
    if (c.size() != n) {
        throw std::runtime_error("Result vector must have same size with a and b in mul_loop!");
    }
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}
/**
 * @brief Converts a real-valued vector to a complex-valued vector.
 *
 * Constructs a complex vector whose real part is given by the input
 * vector and whose imaginary part is zero:
 *
 *     real_vec[i] -> real_vec[i] + 0i
 *
 * @param real_vec Input real-valued vector.
 *
 * @return A complex-valued vector with identical real components and
 *         zero imaginary components.
 */
inline std::vector<std::complex<double>>
double_to_complex_real(const std::vector<double>& real_vec)
{
    std::vector<std::complex<double>> complex_vec;
    complex_vec.reserve(real_vec.size());

    for (double val : real_vec)
    {
        complex_vec.emplace_back(val, 0.0);
    }

    return complex_vec;
}
/**
 * @brief Converts a real-valued vector to a purely imaginary complex vector.
 *
 * Constructs a complex vector whose imaginary part is given by the input
 * vector and whose real part is zero:
 *
 *     real_vec[i] -> 0 + real_vec[i] i
 *
 * @param real_vec Input real-valued vector.
 *
 * @return A complex-valued vector with zero real components and
 *         imaginary components equal to the input values.
 */
inline std::vector<std::complex<double>>
double_to_complex_imag(const std::vector<double>& real_vec)
{
    std::vector<std::complex<double>> complex_vec;
    complex_vec.reserve(real_vec.size());

    for (double val : real_vec)
    {
        complex_vec.emplace_back(0.0, val);
    }

    return complex_vec;
}
/**
 * @brief Extracts the real part of a complex-valued vector.
 *
 * Given a complex vector
 *
 *     z[i] = a[i] + b[i] i
 *
 * this function returns the vector
 *
 *     a[i].
 *
 * @param complex_vec Input complex-valued vector.
 *
 * @return A real-valued vector containing the real component of each
 *         complex element.
 */
inline std::vector<double>
complex_real_to_double(const std::vector<std::complex<double>>& complex_vec)
{
    std::vector<double> real_vec;
    real_vec.reserve(complex_vec.size());

    for (const auto& c : complex_vec)
    {
        real_vec.push_back(c.real());
    }

    return real_vec;
}
/**
 * @brief Extracts the imaginary part of a complex-valued vector.
 *
 * Given a complex vector
 *
 *     z[i] = a[i] + b[i] i
 *
 * this function returns the vector
 *
 *     b[i].
 *
 * @param complex_vec Input complex-valued vector.
 *
 * @return A real-valued vector containing the imaginary component of
 *         each complex element.
 */
inline std::vector<double>
complex_imag_to_double(const std::vector<std::complex<double>>& complex_vec)
{
    std::vector<double> imag_vec;
    imag_vec.reserve(complex_vec.size());

    for (const auto& c : complex_vec)
    {
        imag_vec.push_back(c.imag());
    }

    return imag_vec;
}
/**
 * @brief Generates evenly spaced points over a specified interval.
 *
 * Similar to NumPy's linspace().
 *
 * If endpoint is true, the generated sequence includes both
 * start and end values:
 *
 *     [start, ..., end]
 *
 * Otherwise, the sequence excludes the endpoint:
 *
 *     [start, ..., end)
 *
 * @param start Starting value of the interval.
 * @param end Ending value of the interval.
 * @param num Number of points to generate.
 * @param endpoint Whether the endpoint is included, default is false.
 *
 * @return Vector containing uniformly spaced values.
 *
 * @note If num == 1, the returned vector contains only start or end(if endpoint).
 */
inline std::vector<double> linspace(double start, double end, int num, bool endpoint) {
    if (num<=0) 
        return {};
    if (num == 1) {
        if (endpoint) return {end};
        else return {start};
    }
    std::vector<double> points(num);
    double step;

    if (endpoint) {
        step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i)
            points[i] = start + i * step;
    }
    else {
        step = (end - start) / num;
        for (int i = 0; i < num; ++i)
            points[i] = start + i * step;
    }
    return points;
}
// overload function to support num(size_t).
inline std::vector<double> linspace(double start, double end, size_t num, bool endpoint) {
    if (num == 0) return {};
    if (num == 1) {
        if (endpoint) return {end};
        else return {start};
    }
    std::vector<double> points(num);
    double step;
    if (endpoint) {
        step = (end - start) / static_cast<double>(num - 1);
        for (size_t i = 0; i < num; ++i)
            points[i] = start + i * step;
    } else {
        step = (end - start) / static_cast<double>(num);
        for (size_t i = 0; i < num; ++i)
            points[i] = start + i * step;
    }
    return points;
}
/**
 * @brief Generates the frequency bins associated with a discrete Fourier transform.
 *
 * Equivalent to NumPy's `fftfreq()` and compatible with FFTW's output ordering.
 *
 * For a signal containing @p n samples with sampling interval @p d,
 * the returned frequencies are
 *
 *     f[k] = k / (n d)
 *
 * for the non-negative frequency components, followed by the
 * corresponding negative frequencies:
 *
 * Even n:
 *
 *     [0, 1, ..., n/2-1, -n/2, ..., -1] / (n d)
 *
 * Odd n:
 *
 *     [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (n d)
 *
 * Example:
 *
 *     fftfreq(8, 0.1)
 *
 * returns
 *
 *     [0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25]
 *
 * @param n Number of samples in the signal.
 * @param d Sampling interval (seconds between samples).
 *
 * @return Frequency bins in Hz.
 *
 * @throws std::invalid_argument if n == 0 or d <= 0.
 *
 * @note The returned vector has the same length as the input signal.
 * @note The frequency ordering matches FFTW and NumPy conventions.
 */
inline std::vector<double> fftfreq(std::size_t n, double d)
{
    if (n == 0)
    {
        throw std::invalid_argument(
            "fftfreq: n must be greater than zero."
        );
    }

    if (d <= 0.0)
    {
        throw std::invalid_argument(
            "fftfreq: sampling interval d must be positive."
        );
    }

    std::vector<double> freq(n);

    const double scale = 1.0 / (static_cast<double>(n) * d);
    const std::size_t cutoff = (n + 1) / 2;

    for (std::size_t i = 0; i < n; ++i)
    {
        const long long k =
            (i < cutoff)
            ? static_cast<long long>(i)
            : static_cast<long long>(i) - static_cast<long long>(n);

        freq[i] = static_cast<double>(k) * scale;
    }

    return freq;
}
// overload to process input n with kind "int"
inline std::vector<double> fftfreq(int n, double d)
{
    if (n <= 0)
    {
        throw std::invalid_argument(
            "fftfreq: n must be positive."
        );
    }

    return fftfreq(
        static_cast<std::size_t>(n),
        d
    );
}

// declare useful functions
std::pair<double, double> compute_l2_error(const std::vector<double>& orig, const std::vector<double>& rec);
std::pair<double, double> compute_l2_error(const std::vector<std::complex<double>>& orig, const std::vector<std::complex<double>>& rec);

// 均匀网格上的数值积分工具
// 支持 std::vector<double> / std::vector<std::complex<double>> 等待积序列
// 积分方法
enum class QuadMethod {
    Rectangle,     // 左矩形法,                 O(h)
    Trapezoidal,   // 复化梯形法,               O(h^2)
    Simpson13,     // 复化 Simpson 1/3 (3点),   O(h^4)
    Simpson38,     // 复化 Simpson 3/8 (4点),   O(h^4)
    Boole,         // 复化 Boole  (5点),        O(h^6)
    NC7,           // 7点 Newton-Cotes,         O(h^8)
    PeriodicTrap,  // 周期梯形 (等权求和), 对光滑周期函数具谱精度
    Auto           // 自动选择当前 N 下可用的最高阶方法
};

// 通用复化 Newton-Cotes:
//   w 为归一化权重(含 h 的除法), w.size()-1 为每个区段的区间数 m
//   要求 (N-1) % m == 0
template <typename T>
T integrate_newton_cotes(const std::vector<T>& f, double dx,
                         const std::vector<double>& w)
{
    const size_t N = f.size();
    const size_t m = w.size() - 1;
    if (N < 2) {
        throw std::invalid_argument("integrate: need at least 2 samples");
    }
    if ((N - 1) % m != 0) {
        throw std::invalid_argument(
            "integrate: Simpson/Boole 等方法要求 N-1 能被 "
            + std::to_string(m) + " 整除 (当前 N=" + std::to_string(N) + ")");
    }
    T sum{};   // double -> 0.0, complex -> (0,0)
    for (size_t b = 0; b < N - 1; b += m) {
        for (size_t j = 0; j <= m; ++j) {
            sum += w[j] * f[b + j];
        }
    }
    return dx * sum;
}


// 主入口: 对均匀网格采样序列 f (间距 dx) 积分
template <typename T>
T integrate(const std::vector<T>& f, double dx,
            QuadMethod method = QuadMethod::Auto)
{
    const size_t N = f.size();
    if (N < 2) {
        throw std::invalid_argument("integrate: need at least 2 samples");
    }
    switch (method) {

    case QuadMethod::Rectangle: {           // 左矩形
        T sum{};
        for (size_t i = 0; i + 1 < N; ++i) sum += f[i];
        return dx * sum;
    }

    case QuadMethod::Trapezoidal: {         // 复化梯形
        T sum = 0.5 * (f.front() + f.back());
        for (size_t i = 1; i + 1 < N; ++i) sum += f[i];
        return dx * sum;
    }

    case QuadMethod::PeriodicTrap: {        // 周期梯形: 等权
        T sum{};
        for (const auto& v : f) sum += v;
        return dx * sum;
    }

    // ---- 高阶 Newton-Cotes (权重已归一化) ----
    case QuadMethod::Simpson13:
        return integrate_newton_cotes<T>(f, dx, {1.0/3.0, 4.0/3.0, 1.0/3.0});
    case QuadMethod::Simpson38:
        return integrate_newton_cotes<T>(f, dx, {3.0/8.0, 9.0/8.0, 9.0/8.0, 3.0/8.0});
    case QuadMethod::Boole:
        return integrate_newton_cotes<T>(f, dx,
            {14.0/45.0, 64.0/45.0, 24.0/45.0, 64.0/45.0, 14.0/45.0});
    case QuadMethod::NC7:
        return integrate_newton_cotes<T>(f, dx,
            {41.0/140.0, 216.0/140.0, 27.0/140.0, 272.0/140.0,
             27.0/140.0, 216.0/140.0, 41.0/140.0});

    case QuadMethod::Auto:   // 从高到低, 选当前 N 可用的最高阶
        // m = 权重个数 - 1 = 每个区段的区间数
        if (N >= 7 && (N - 1) % 6 == 0) return integrate(f, dx, QuadMethod::NC7);     // 7点, m=6
        if (N >= 5 && (N - 1) % 4 == 0) return integrate(f, dx, QuadMethod::Boole);   // 5点, m=4
        if (N >= 4 && (N - 1) % 3 == 0) return integrate(f, dx, QuadMethod::Simpson38);
        if (N >= 3 && (N - 1) % 2 == 0) return integrate(f, dx, QuadMethod::Simpson13);
        return integrate(f, dx, QuadMethod::Trapezoidal);
    }
    throw std::invalid_argument("integrate: unknown QuadMethod");
}

//fft tools
/**
 * @brief Reusable one-dimensional complex FFT operator.
 *
 * Creates FFTW forward and inverse plans once and
 * reuses them for all subsequent transforms of
 * the same size.
 *
 * Forward transform:
 *
 *     X(k) = Σ x(n) exp(-i2πkn/N)
 *
 * Inverse transform:
 *
 *     x(n) = (1/N) Σ X(k) exp(+i2πkn/N)
 *
 * The inverse transform is automatically normalized
 * by 1/N.
 *
 * @note Returned references refer to internal buffers
 *       that are overwritten by the next transform.
 *
 * @warning This class is not thread-safe.
 */
class fft1d
{
private:

    size_t N_;

    std::vector<std::complex<double>> in_buffer_;

    std::vector<std::complex<double>> out_buffer_;

    fftw_plan forward_plan_;

    fftw_plan backward_plan_;

public:

    explicit fft1d(size_t N);

    ~fft1d();

    fft1d(const fft1d&) = delete;

    fft1d&
    operator=(const fft1d&) = delete;

    fft1d(fft1d&&) = delete;
    fft1d& operator=(fft1d&&) = delete;

    const std::vector<std::complex<double>>&
    fft(const std::vector<std::complex<double>>& input);

    const std::vector<std::complex<double>>&
    ifft(const std::vector<std::complex<double>>& input);

    

    size_t size() const noexcept
    {
        return N_;
    }
};

// 2Dfft
/**
 * @brief Reusable two-dimensional complex FFT operator.
 *
 * Creates FFTW forward and inverse plans once and
 * reuses them for all subsequent transforms of
 * the same size.
 *
 * Input and output are flattened arrays in row‑major order:
 * index = x * Ny + y, where x = 0..Nx-1, y = 0..Ny-1.
 *
 * Forward transform:
 *
 *     X(u,v) = Σ_x Σ_y f(x,y) exp(-i2π(ux/Nx + vy/Ny))
 *
 * Inverse transform:
 *
 *     f(x,y) = (1/(Nx*Ny)) Σ_u Σ_v X(u,v) exp(+i2π(ux/Nx + vy/Ny))
 *
 * The inverse transform is automatically normalized by 1/(Nx*Ny).
 *
 * @note Returned references refer to internal buffers
 *       that are overwritten by the next transform.
 *
 * @warning This class is not thread-safe.
 */
class fft2d
{
private:
    size_t Nx_;
    size_t Ny_;
    size_t total_;

    std::vector<std::complex<double>> in_buffer_;
    std::vector<std::complex<double>> out_buffer_;

    fftw_plan forward_plan_;
    fftw_plan backward_plan_;

public:
    /**
     * @brief Construct a 2D FFT object for size Nx × Ny.
     * @param Nx Number of rows (x dimension).
     * @param Ny Number of columns (y dimension).
     * @throws std::invalid_argument if Nx <= 0 or Ny <= 0.
     * @throws std::runtime_error if FFTW plan creation fails.
     */
    explicit fft2d(size_t Nx, size_t Ny);

    ~fft2d();

    fft2d(const fft2d&) = delete;
    fft2d& operator=(const fft2d&) = delete;
    fft2d(fft2d&&) = delete;
    fft2d& operator=(fft2d&&) = delete;

    /**
     * @brief Perform forward 2D FFT.
     * @param input Flattened input of size Nx*Ny (row‑major order).
     * @return Reference to internal buffer containing the transform result.
     * @throws std::invalid_argument if input size does not match Nx*Ny.
     */
    const std::vector<std::complex<double>>&
    fft2(const std::vector<std::complex<double>>& input);

    /**
     * @brief Perform inverse 2D FFT (normalized).
     * @param input Flattened input of size Nx*Ny (row‑major order).
     * @return Reference to internal buffer containing the inverse result.
     * @throws std::invalid_argument if input size does not match Nx*Ny.
     */
    const std::vector<std::complex<double>>&
    ifft2(const std::vector<std::complex<double>>& input);

    /// @return Number of rows (x dimension).
    size_t nx() const noexcept { return Nx_; }

    /// @return Number of columns (y dimension).
    size_t ny() const noexcept { return Ny_; }

    /// @return Total number of elements (Nx * Ny).
    size_t size() const noexcept { return total_; }
};

class fft_op {
public:
    // 1d constructor
    fft_op(size_t N) : dim(1), impl(std::in_place_type<fft1d>, N) {}
    // 2d constructor
    fft_op(size_t Nx, size_t Ny) : dim(2), impl(std::in_place_type<fft2d>, Nx, Ny) {}

    fft_op(const fft_op&) = delete;
    fft_op& operator=(const fft_op&) = delete;
    fft_op(fft_op&&) = delete;
    fft_op& operator=(fft_op&&) = delete;

    const std::vector<std::complex<double>>&
    fft(const std::vector<std::complex<double>>& input) {
        return std::visit([&input](auto& obj) -> const std::vector<std::complex<double>>& {
            using T = std::decay_t<decltype(obj)>;
            if constexpr (std::is_same_v<T, fft1d>) {
                return obj.fft(input);
            } else if constexpr (std::is_same_v<T, fft2d>) {
                return obj.fft2(input);
            }
            else throw std::logic_error("invalid fft dimension");
        }, impl);
    }

    const std::vector<std::complex<double>>&
    ifft(const std::vector<std::complex<double>>& input) {
        return std::visit([&input](auto& obj) -> const std::vector<std::complex<double>>& {
            using T = std::decay_t<decltype(obj)>;
            if constexpr (std::is_same_v<T, fft1d>) {
                return obj.ifft(input);
            } else if constexpr (std::is_same_v<T, fft2d>) {
                return obj.ifft2(input);
            }
            else throw std::logic_error("invalid fft dimension");
        }, impl);
    }

private:
    int dim;
    std::variant<fft1d, fft2d> impl;
};

// global fft_op object - external initialization required
extern std::unique_ptr<fft_op> fft_obj;

// =====================================================================
// Low-pass filter tools
// =====================================================================

/**
 * @brief Available low-pass filter types.
 *
 * All filters accept a cut-off wavenumber k_cut and act on the
 * magnitude |k| of each wavenumber in k_axis. Components with
 * |k| <= k_cut are (approximately) preserved, components with
 * |k| > k_cut are attenuated.
 */
enum class LowPassType {
    Ideal,        ///< Sharply truncates |k| > k_cut.        (Gibbs ringing)
    Exponential,  ///< exp(-(|k|/k_cut)^order).             (Very smooth)
    Gaussian,     ///< exp(-0.5*(|k|/sigma)^2).             (Smooth)
    FivePoint,    ///< Five-point 
    Butterworth   ///< 1 / sqrt(1 + (|k|/k_cut)^(2*order)).
};

/**
 * @brief Builds a real-valued low-pass transfer function H(k).
 *
 * The returned vector has the same size as @p k_axis and contains
 * the multiplicative filter weights applied to the spectrum.
 *
 * @param k_axis Wavenumber axis. Values may be negative (FFT order).
 * @param k_cut  Cut-off wavenumber (>0).
 * @param type   Filter type (see LowPassType).
 * @param width  Transition half-width used by Gaussian / FivePoint /
 *               Exponential filters. Interpreted as a fraction of k_cut
 *               when type == FivePoint (0..1). For Gaussian / Exponential
 *               it is a relative scale factor.
 * @param order  Filter order for Exponential / Butterworth.
 *
 * @return Vector<double> H(k) of the same length as k_axis.
 *
 * @throws std::invalid_argument if k_cut <= 0, k_axis empty,
 *         or width <= 0 for smooth filters.
 */
std::vector<double> build_lowpass_transfer(
    const std::vector<double>& k_axis,
    double k_cut,
    LowPassType type,
    double width = 1.0,
    double order = 2.0);

/**
 * @brief Applies a low-pass filter to a complex spectrum.
 *
 * Computes
 *
 *     f_hat_filtered[i] = H(k_axis[i]) * f_hat[i]
 *
 * and returns the filtered spectrum. The input vectors are not modified.
 *
 * @param k_axis Wavenumber axis (same size as f_hat).
 * @param f_hat  Input complex spectrum.
 * @param k_cut  Cut-off wavenumber.
 * @param type   Filter type.
 * @param width  Transition width (see build_lowpass_transfer).
 * @param order  Filter order (see build_lowpass_transfer).
 *
 * @return Filtered complex spectrum, same size as f_hat.
 *
 * @throws std::runtime_error if k_axis.size() != f_hat.size().
 */
std::vector<std::complex<double>> lowpass_filter_spectrum(
    const std::vector<double>& k_axis,
    const std::vector<std::complex<double>>& f_hat,
    double k_cut,
    LowPassType type,
    double width = 1.0,
    double order = 2.0);

#endif//VAR_OP_FUNCTION_H