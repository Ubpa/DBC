cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(DBC VERSION 1.0.0)
message(STATUS "[Project] ${PROJECT_NAME}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CustomInit)

Custom_InitProject()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

function(Custom_TorchCopyDll TargetName)
  # The following code block is suggested to be used on Windows.
  # According to https://github.com/pytorch/pytorch/issues/25457,
  # the DLLs need to be copied to avoid memory errors.
  if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET "${TargetName}"
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif (MSVC)
endfunction()

Custom_AddSubDirsRec(src)
