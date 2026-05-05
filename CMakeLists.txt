cmake_minimum_required(VERSION 3.15...3.30)
project(vtlengine_cpp_parser LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------------------
# ANTLR4 C++ runtime
# -------------------------------------------------------------------
# Two modes:
#   * default — compile the vendored sources in
#     third_party/antlr4-cpp-runtime/src into a static library.
#   * prebuilt — link against a previously-built static library + headers
#     under VTLENGINE_ANTLR4_PREFIX. Used in CI to avoid recompiling 138
#     third-party translation units on every wheel build.
option(VTLENGINE_USE_PREBUILT_ANTLR4 "Link against a prebuilt ANTLR4 runtime" OFF)
option(VTLENGINE_BUILD_ANTLR4_PREBUILT "Install antlr4_runtime + headers (for prebuilt artifact builds)" OFF)
set(VTLENGINE_ANTLR4_PREFIX "" CACHE PATH "Prefix containing lib/ and include/ for prebuilt ANTLR4 runtime")

set(ANTLR4_SRC_DIR "${CMAKE_SOURCE_DIR}/third_party/antlr4-cpp-runtime/src")

if(VTLENGINE_USE_PREBUILT_ANTLR4)
    if(NOT VTLENGINE_ANTLR4_PREFIX)
        message(FATAL_ERROR "VTLENGINE_USE_PREBUILT_ANTLR4=ON requires VTLENGINE_ANTLR4_PREFIX")
    endif()
    find_library(ANTLR4_LIB NAMES antlr4_runtime
        PATHS "${VTLENGINE_ANTLR4_PREFIX}/lib"
        REQUIRED NO_DEFAULT_PATH)
    add_library(antlr4_runtime STATIC IMPORTED)
    set_target_properties(antlr4_runtime PROPERTIES
        IMPORTED_LOCATION "${ANTLR4_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${VTLENGINE_ANTLR4_PREFIX}/include"
        INTERFACE_COMPILE_DEFINITIONS ANTLR4CPP_STATIC)
else()
    file(GLOB_RECURSE ANTLR4_RUNTIME_SOURCES "${ANTLR4_SRC_DIR}/*.cpp")
    add_library(antlr4_runtime STATIC ${ANTLR4_RUNTIME_SOURCES})
    target_include_directories(antlr4_runtime PUBLIC "${ANTLR4_SRC_DIR}")
    target_compile_definitions(antlr4_runtime PUBLIC ANTLR4CPP_STATIC)
    if(MSVC)
        target_compile_options(antlr4_runtime PRIVATE /w)
        # Fix missing chrono include in ProfilingATNSimulator on MSVC
        target_compile_options(antlr4_runtime PRIVATE /FI"chrono")
    else()
        target_compile_options(antlr4_runtime PRIVATE -w)
    endif()
    if(VTLENGINE_BUILD_ANTLR4_PREBUILT)
        install(TARGETS antlr4_runtime DESTINATION lib)
        install(DIRECTORY "${ANTLR4_SRC_DIR}/" DESTINATION include FILES_MATCHING PATTERN "*.h")
    endif()
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
# Skipped when only producing the prebuilt-runtime artifact for caching:
# the standalone runtime build does not need pybind11.
if(NOT VTLENGINE_BUILD_ANTLR4_PREBUILT)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(vtl_cpp_parser
        "${VTL_PARSER_DIR}/bindings.cpp"
        ${VTL_PARSER_SOURCES}
    )

    # ANTLR4 includes are propagated via the antlr4_runtime target's
    # INTERFACE_INCLUDE_DIRECTORIES (PUBLIC for the from-source build, set
    # explicitly on the imported target for the prebuilt build).
    target_include_directories(vtl_cpp_parser PRIVATE "${VTL_PARSER_DIR}")

    target_link_libraries(vtl_cpp_parser PRIVATE antlr4_runtime)

    # Suppress warnings from generated parser code
    if(MSVC)
        target_compile_options(vtl_cpp_parser PRIVATE /w)
    else()
        target_compile_options(vtl_cpp_parser PRIVATE -w)
    endif()

    # Install the module into the vtlengine._cpp_parser package
    install(TARGETS vtl_cpp_parser DESTINATION vtlengine/AST/Grammar/_cpp_parser)
endif()
