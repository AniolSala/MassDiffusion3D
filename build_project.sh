#!/bin/sh

python_executable="python3"
run_base_tests=false
run_corrections_test=false

# Check python executable
python3_path=$(command -v "$python_executable")
if [ -z "$python3_path" ]; then
	echo "Error: python executable not found: $python_executable" >&2
	exit 1
fi
echo "Compiling project using python from $python3_path"

# Build project (roots / norms table names come from the CMake cache defaults
# at CMakeLists.txt:8-9; pass -DSTRATIFIED_ROOTS_FILE_NAME=... -DNORMS_FILE_NAME=...
# to cmake directly if you want a different table).
cmake -S . -B build \
	-DPYTHON_EXECUTABLE:FILEPATH="$python3_path" && cmake --build build


if [ "$run_base_tests" = true ] && [ "$run_corrections_test" = true ]; then
	ctest --test-dir build --output-on-failure
elif [ "$run_base_tests" = true ]; then
	ctest --test-dir build -R '^base_tests$' --output-on-failure
elif [ "$run_corrections_test" = true ]; then
	ctest --test-dir build -R '^finite_peclet_corrections_tests$' --output-on-failure
fi
