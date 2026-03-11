cmake_minimum_required(VERSION 3.15...3.30)
project(vtlengine_cpp_parser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# -------------------------------------------------------------------
# ANTLR4 C++ runtime (vendored as static library)
# -------------------------------------------------------------------
set(ANTLR4_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/antlr4-cpp-runtime/src")

# Collect all ANTLR4 runtime .cpp files
file(GLOB_RECURSE ANTLR4_RUNTIME_SOURCES "${ANTLR4_SRC_DIR}/*.cpp")

# Exclude profiling (not needed, has MSVC chrono issues)
list(FILTER ANTLR4_RUNTIME_SOURCES EXCLUDE REGEX ".*ProfilingATNSimulator\\.cpp$")

add_library(antlr4_runtime STATIC ${ANTLR4_RUNTIME_SOURCES})
target_include_directories(antlr4_runtime PUBLIC "${ANTLR4_SRC_DIR}")
target_compile_definitions(antlr4_runtime PUBLIC ANTLR4CPP_STATIC)

# Suppress warnings from third-party code
if(MSVC)
    target_compile_options(antlr4_runtime PRIVATE /w)
else()
    target_compile_options(antlr4_runtime PRIVATE -w)
endif()

# -------------------------------------------------------------------
# Generated VTL parser (C++ files from ANTLR)
# -------------------------------------------------------------------
set(VTL_PARSER_DIR "${CMAKE_SOURCE_DIR}/src/vtlengine/AST/Grammar/_cpp_parser")

set(VTL_PARSER_SOURCES
    "${VTL_PARSER_DIR}/VtlLexer.cpp"
    "${VTL_PARSER_DIR}/VtlParser.cpp"
    "${VTL_PARSER_DIR}/VtlBaseListener.cpp"
    "${VTL_PARSER_DIR}/VtlBaseVisitor.cpp"
    "${VTL_PARSER_DIR}/VtlListener.cpp"
    "${VTL_PARSER_DIR}/VtlVisitor.cpp"
)

# -------------------------------------------------------------------
# pybind11 module
# -------------------------------------------------------------------
pybind11_add_module(vtl_cpp_parser
    "${VTL_PARSER_DIR}/bindings.cpp"
    ${VTL_PARSER_SOURCES}
)

target_include_directories(vtl_cpp_parser PRIVATE
    "${ANTLR4_SRC_DIR}"
    "${VTL_PARSER_DIR}"
)

target_link_libraries(vtl_cpp_parser PRIVATE antlr4_runtime)

# Suppress warnings from generated parser code
if(MSVC)
    target_compile_options(vtl_cpp_parser PRIVATE /w)
else()
    target_compile_options(vtl_cpp_parser PRIVATE -w)
endif()

# Install the module into the vtlengine._cpp_parser package
install(TARGETS vtl_cpp_parser DESTINATION vtlengine/AST/Grammar/_cpp_parser)
