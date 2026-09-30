# CMake generated Testfile for 
# Source directory: /home/aniol/repos/AS
# Build directory: /home/aniol/repos/AS/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(base_tests "/home/aniol/repos/AS/bin/cpp_base_tests")
set_tests_properties(base_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/aniol/repos/AS/CMakeLists.txt;151;add_test;/home/aniol/repos/AS/CMakeLists.txt;0;")
subdirs("extern/pybind11")
