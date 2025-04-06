set(mode "")
set(private_defines "")
if(Ubpa_UMBD_UsingSharedLibrary)
  set(mode SHARED)
  list(APPEND private_defines "UBPA_UMBD_BUILDING_SHARED_LIBRARY")
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

Ubpa_AddTarget(
  MODE ${mode}
  CXX_STANDARD 17
  PRIVATE_DEFINE
    ${private_defines}
  SOURCE
    "${PROJECT_SOURCE_DIR}/include/NeuralAidedMBD"
  INC
    "${PROJECT_SOURCE_DIR}/include"
  LIB
    Ubpa::UMBD_CeresSolver
	Ubpa::UMBD_IO
    Ubpa::CMakeLibTorch_bc7e
    "${TORCH_LIBRARIES}"
  LIB_PRIVATE
    Ceres::ceres
  C_OPTION_PRIVATE
    ${c_options_private}
)
