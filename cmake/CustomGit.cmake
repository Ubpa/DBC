# ----------------------------------------------------------------------------
#
# Custom_InitGit()
# - find git [required]
#
# ----------------------------------------------------------------------------
#
# Custom_UpdateSubModule()
# - update submodule
#
# ----------------------------------------------------------------------------

message(STATUS "include CustomGit.cmake")

macro(Custom_InitGit)
  message(STATUS "----------")
  find_package(Git REQUIRED)
  message(STATUS "GIT_FOUND: ${GIT_FOUND}")
  message(STATUS "GIT_EXECUTABLE: ${GIT_EXECUTABLE}")
  message(STATUS "GIT_VERSION_STRING: ${GIT_VERSION_STRING}")
endmacro()

function(Custom_UpdateSubModule)
  if(NOT GIT_FOUND)
    message(FATAL_ERROR "you should call Custom_InitGit() before calling Custom_UpdateSubModule()")
  endif()
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule init
    #OUTPUT_VARIABLE out
    #OUTPUT_STRIP_TRAILING_WHITESPACE
    #ERROR_QUIET
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update
    #OUTPUT_VARIABLE out
    #OUTPUT_STRIP_TRAILING_WHITESPACE
    #ERROR_QUIET
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
endfunction()
