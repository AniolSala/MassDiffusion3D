cmake_minimum_required(VERSION 3.10)
project(MassDiffusion3DPlus CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Stratified roots/norms tables (inside src/CDStratifiedSolution/data). Changing
# these defaults also updates existing build directories on their next configure.
# Override with -DSTRATIFIED_ROOTS_FILE_NAME=... -DNORMS_FILE_NAME=...; an
# override is kept until you pass an empty value to return to the default.
set(CDS_DEFAULT_ROOTS_FILE "roots_200x200.txt")
set(CDS_DEFAULT_NORMS_FILE "norms_200x200_cpp.txt")

# A plain set(... CACHE ...) never replaces a value already in CMakeCache.txt, so
# build directories configured earlier would keep an outdated default. Remember
# the default each value came from and refresh the value unless it was overridden.
function(cds_data_file_option var default legacy_default doc)
    if(DEFINED _${var}_DEFAULT)
        set(previous_default "${_${var}_DEFAULT}")
    elseif(EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
        # Build directory configured before defaults were tracked.
        set(previous_default "${legacy_default}")
    else()
        set(previous_default "")
    endif()
    if("${${var}}" STREQUAL "" OR "${${var}}" STREQUAL "${previous_default}")
        set(${var} "${default}" CACHE STRING "${doc}" FORCE)
    endif()
    set(_${var}_DEFAULT "${default}" CACHE INTERNAL "Default ${var} was last configured with")
endfunction()

cds_data_file_option(STRATIFIED_ROOTS_FILE_NAME "${CDS_DEFAULT_ROOTS_FILE}" "roots_100x100.txt"
    "Stratified roots data filename inside src/CDStratifiedSolution/data")
cds_data_file_option(NORMS_FILE_NAME "${CDS_DEFAULT_NORMS_FILE}" "norms_100x100_cpp.txt"
    "Norms data filename inside src/CDStratifiedSolution/data")
add_compile_definitions(
    CDS_ROOTS_FILE="${STRATIFIED_ROOTS_FILE_NAME}"
    CDS_NORMS_FILE="${NORMS_FILE_NAME}"
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)

add_link_options("-flto=auto")

# Add Pybind11
add_subdirectory(extern/pybind11)

# Sources
file(GLOB SOURCES "src/*.cpp")  # All cpp files
set(CORE_SOURCES ${SOURCES})
set(PYBIND_SOURCES ${SOURCES})
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/pybind.cpp")
list(REMOVE_ITEM CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
list(REMOVE_ITEM PYBIND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")  # Remove main.cpp

# Enable OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found!")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found.")
endif()

# Find CGAL
# find_package(CGAL REQUIRED)
# if(CGAL_FOUND)
#     message(STATUS "CGAL found!")
# else()
#     message(FATAL_ERROR "CGAL not found.")
# endif()

# Create main executable
add_executable(main ${CORE_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX CGAL::CGAL)

# Find Python (choose the version explicitly)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Create Python module
# message("SOURCES = ${SOURCES}")
# message("EXEC_SOURCES = ${EXEC_SOURCES}")
# message("PYBIND_SOURCES = ${PYBIND_SOURCES}")

pybind11_add_module(MassDiffusion3Dcpp ${PYBIND_SOURCES})
target_include_directories(MassDiffusion3Dcpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(MassDiffusion3Dcpp PRIVATE OpenMP::OpenMP_CXX)  # Enable OpenMP
# target_link_libraries(MassDiffusion3Dcpp PRIVATE OpenMP::OpenMP_CXX CGAL::CGAL)  # Enable OpenMP and CGAL for Python module

# C++ unit tests
enable_testing()

# Unit-test sources for the base solution.
set(BASE_TEST_SOURCES
    tests/test_mode_cutoff.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_psinm_r.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gaussian_integration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compute_coefficients.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parseval_identity.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flux_calculation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_solution_at_planes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_solution_at_planes_variable_points.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_group_points_in_planes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_get_solution_cloud_of_points.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_coeff_zero_identity_highm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_dpsinm_r_2_underflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graetz_eigenvalues.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graetz_coefficients.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_graetz_solution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_nusselt_number.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_set_get_max_root.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pseudoproduct_measure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finite_peclet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finite_peclet_radial_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finite_peclet_coefficients_gaussian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finite_peclet_coefficients_gauss_jacobi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_finite_peclet_norms_boundary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qep.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_gram_backends.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_inlet_projection_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blurriness.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_gram_node_rule.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_rhs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shifted_jacobi_basis.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_rhs_representer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_rhs_cross_validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_radial_projection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fp_representer_radial.cpp
)

# Base test binary
add_executable(cpp_base_tests
    ${CORE_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/tinytest_main.cpp
    ${BASE_TEST_SOURCES}
)
target_include_directories(cpp_base_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
)
target_compile_definitions(cpp_base_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(cpp_base_tests PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME base_tests COMMAND cpp_base_tests)
