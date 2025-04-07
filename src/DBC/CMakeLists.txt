set(private_defines "")

set(c_options_private "")
if(MSVC)
  list(APPEND c_options_private "/MP")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  #
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  #
endif()

Custom_AddTarget(
  MODE STATIC
  CXX_STANDARD 17
  PRIVATE_DEFINE
    ${private_defines}
  SOURCE
    "${PROJECT_SOURCE_DIR}/include/DBC"
  INC
    "${PROJECT_SOURCE_DIR}/include"
  LIB
    "${TORCH_LIBRARIES}"
  C_OPTION_PRIVATE
    ${c_options_private}
)
