# Minimal C project that links the Oxideon C API (`include/oxideon.h`).
#
#   cmake -S examples/c_embed -B build/c_embed
#   cmake --build build/c_embed --config Release
#   ./build/c_embed/c_embed            (build/c_embed/Release/c_embed.exe on Windows)
#
# The shared library is built with `cargo build --release -p oxideon` (the facade crate is a `cdylib`, and it
# re-exports the `oxideon_*` C symbols from `oxideon-core`), so a Rust toolchain must be on PATH.
cmake_minimum_required(VERSION 3.15)
project(oxideon_c_embed C)

set(OXIDEON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(OXIDEON_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
set(OXIDEON_LIB_DIR "${OXIDEON_CARGO_TARGET_DIR}/release")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

if(WIN32)
  set(OXIDEON_LIB_FILE "${OXIDEON_LIB_DIR}/oxideon.dll")
  set(OXIDEON_IMPLIB_FILE "${OXIDEON_LIB_DIR}/oxideon.dll.lib")
  set(OXIDEON_BYPRODUCTS "${OXIDEON_LIB_FILE}" "${OXIDEON_IMPLIB_FILE}")
elseif(APPLE)
  set(OXIDEON_LIB_FILE "${OXIDEON_LIB_DIR}/liboxideon.dylib")
  set(OXIDEON_BYPRODUCTS "${OXIDEON_LIB_FILE}")
else()
  set(OXIDEON_LIB_FILE "${OXIDEON_LIB_DIR}/liboxideon.so")
  set(OXIDEON_BYPRODUCTS "${OXIDEON_LIB_FILE}")
endif()

add_custom_target(oxideon_rust ALL
  COMMAND "${CARGO_EXECUTABLE}" build --release -p oxideon
          --manifest-path "${OXIDEON_ROOT}/Cargo.toml"
          --target-dir "${OXIDEON_CARGO_TARGET_DIR}"
  BYPRODUCTS ${OXIDEON_BYPRODUCTS}
  COMMENT "Building the Oxideon shared library with cargo"
  USES_TERMINAL)

add_library(oxideon SHARED IMPORTED GLOBAL)
set_target_properties(oxideon PROPERTIES
  IMPORTED_LOCATION "${OXIDEON_LIB_FILE}"
  INTERFACE_INCLUDE_DIRECTORIES "${OXIDEON_ROOT}/include")
if(WIN32)
  set_target_properties(oxideon PROPERTIES IMPORTED_IMPLIB "${OXIDEON_IMPLIB_FILE}")
endif()

add_executable(c_embed main.c)
add_dependencies(c_embed oxideon_rust)
target_link_libraries(c_embed PRIVATE oxideon)

if(WIN32)
  # the DLL must sit next to the executable
  add_custom_command(TARGET c_embed POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OXIDEON_LIB_FILE}" "$<TARGET_FILE_DIR:c_embed>")
else()
  set_target_properties(c_embed PROPERTIES BUILD_RPATH "${OXIDEON_LIB_DIR}")
endif()
