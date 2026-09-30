#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tinytest {

struct Failure : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase> &registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline void add(const std::string &name, std::function<void()> fn) {
    registry().push_back(TestCase{name, std::move(fn)});
}

inline void require(bool ok, const char *expr, const char *file, int line) {
    if (!ok) {
        std::ostringstream os;
        os << file << ":" << line << ": REQUIRE(" << expr << ") failed";
        throw Failure(os.str());
    }
}

inline void check(bool ok, const char *expr, const char *file, int line) {
    if (!ok) {
        std::ostringstream os;
        os << file << ":" << line << ": CHECK(" << expr << ") failed";
        throw Failure(os.str());
    }
}

template <typename T>
inline bool approx_equal(T a, T b, T rel, T abs) {
    T diff = std::fabs(a - b);
    if (diff <= abs) {
        return true;
    }
    T denom = std::fabs(b) > abs ? std::fabs(b) : abs;
    return diff <= rel * denom;
}

inline int run_all() {
    int failures = 0;
    for (const auto &tc : registry()) {
        try {
            tc.fn();
            std::cout << "[PASS] " << tc.name << "\n";
        } catch (const Failure &err) {
            failures++;
            std::cout << "[FAIL] " << tc.name << ": " << err.what() << "\n";
        } catch (const std::exception &err) {
            failures++;
            std::cout << "[FAIL] " << tc.name << ": " << err.what() << "\n";
        } catch (...) {
            failures++;
            std::cout << "[FAIL] " << tc.name << ": unknown error\n";
        }
    }
    std::cout << "Ran " << registry().size() << " tests, failures: " << failures << "\n";
    return failures;
}

} // namespace tinytest

#define TEST_CASE(name)                       \
    static void name();                        \
    namespace {                                \
    struct name##_reg {                        \
        name##_reg() {                         \
            tinytest::add(#name, name);        \
        }                                      \
    };                                         \
    static name##_reg name##_reg_instance;     \
    }                                          \
    static void name()

#define REQUIRE(expr) tinytest::require((expr), #expr, __FILE__, __LINE__)
#define CHECK(expr) tinytest::check((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_APPROX(a, b, rel, abs) \
    tinytest::require(tinytest::approx_equal((a), (b), (rel), (abs)), #a " ~= " #b, __FILE__, __LINE__)
#define CHECK_APPROX(a, b, rel, abs) \
    tinytest::check(tinytest::approx_equal((a), (b), (rel), (abs)), #a " ~= " #b, __FILE__, __LINE__)
