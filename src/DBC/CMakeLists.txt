set(mode "")
set(private_defines "")
if(Custom_UMBD_UsingSharedLibrary)
  set(mode SHARED)
  list(APPEND private_defines "Custom_UMBD_BUILDING_SHARED_LIBRARY")
else()
  set(mode STATIC)
endif()

set(c_options_private "")
if(MSVC)
  list(APPEND c_options_private "/MP")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  #
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  #
endif()

Custom_AddTarget(
  MODE ${mode}
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
