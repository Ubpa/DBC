function(Custom_List_Print)
  cmake_parse_arguments("ARG" "" "TITLE;PREFIX" "STRS" ${ARGN})
  list(LENGTH ARG_STRS strsLength)
  if(NOT strsLength)
    return()
  endif()
  if(NOT ${ARG_TITLE} STREQUAL "")
    message(STATUS ${ARG_TITLE})
  endif()
  foreach(str ${ARG_STRS})
    message(STATUS "${ARG_PREFIX}${str}")
  endforeach()
endfunction()

function(Custom_GetDirName dirName)
  string(REGEX MATCH "([^/]*)$" TMP ${CMAKE_CURRENT_SOURCE_DIR})
  set(${dirName} ${TMP} PARENT_SCOPE)
endfunction()

function(Custom_Path_Back rst path times)
  math(EXPR stop "${times}-1")
  set(curPath ${path})
  foreach(index RANGE ${stop})
    string(REGEX MATCH "(.*)/" _ ${curPath})
    set(curPath ${CMAKE_MATCH_1})
  endforeach()
  set(${rst} ${curPath} PARENT_SCOPE)
endfunction()

function(Custom_AddSubDirsRec path)
  file(GLOB_RECURSE children LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/${path}/*)
  set(dirs "")
  list(APPEND children "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
  foreach(item ${children})
    if(IS_DIRECTORY ${item} AND EXISTS "${item}/CMakeLists.txt")
      list(APPEND dirs ${item})
    endif()
  endforeach()
  foreach(dir ${dirs})
    add_subdirectory(${dir})
  endforeach()
endfunction()

function(Custom_GetTargetName rst targetPath)
  file(RELATIVE_PATH targetRelPath "${PROJECT_SOURCE_DIR}/src" "${targetPath}")
  string(REPLACE "/" "_" targetName "${PROJECT_NAME}_${targetRelPath}")
  set(${rst} ${targetName} PARENT_SCOPE)
endfunction()

function(_Custom_ExpandSources rst _sources)
  set(tmp_rst "")
  foreach(item ${${_sources}})
    if(IS_DIRECTORY ${item})
      file(GLOB_RECURSE itemSrcs
        # cmake
        ${item}/*.cmake

        # msvc
        ${item}/*.natvis
        
        # INTERFACEer files
        ${item}/*.h
        ${item}/*.hpp
        ${item}/*.hxx
        ${item}/*.inl
        
        # source files
        ${item}/*.c
        
        ${item}/*.cc
        ${item}/*.cpp
        ${item}/*.cxx
      )
      list(APPEND tmp_rst ${itemSrcs})
    else()
      if(NOT IS_ABSOLUTE "${item}")
		get_filename_component(item "${item}" ABSOLUTE)
      endif()
      list(APPEND tmp_rst ${item})
    endif()
  endforeach()
  set(${rst} ${tmp_rst} PARENT_SCOPE)
endfunction()

function(Custom_AddTarget)
  message(STATUS "----------")

  set(arglist "")
  # public
  list(APPEND arglist SOURCE_PUBLIC INC LIB DEFINE C_OPTION L_OPTION PCH_PUBLIC)
  # interface
  list(APPEND arglist SOURCE_INTERFACE INC_INTERFACE LIB_INTERFACE DEFINE_INTERFACE C_OPTION_INTERFACE L_OPTION_INTERFACE PCH_INTERFACE)
  # private
  list(APPEND arglist SOURCE INC_PRIVATE LIB_PRIVATE DEFINE_PRIVATE C_OPTION_PRIVATE L_OPTION_PRIVATE PCH)
  cmake_parse_arguments(
    "ARG"
    "TEST;NOT_GROUP"
    "MODE;ADD_CURRENT_TO;OUTPUT_NAME;RET_TARGET_NAME;CXX_STANDARD;PCH_REUSE_FROM"
    "${arglist}"
    ${ARGN}
  )
  
  # default
  if("${ARG_ADD_CURRENT_TO}" STREQUAL "")
    set(ARG_ADD_CURRENT_TO "PRIVATE")
  endif()

  # public, private -> interface
  if("${ARG_MODE}" STREQUAL "INTERFACE")
    list(APPEND ARG_SOURCE_INTERFACE   ${ARG_SOURCE_PUBLIC} ${ARG_SOURCE}          )
    list(APPEND ARG_INC_INTERFACE      ${ARG_INC}           ${ARG_INC_PRIVATE}     )
    list(APPEND ARG_LIB_INTERFACE      ${ARG_LIB}           ${ARG_LIB_PRIVATE}     )
    list(APPEND ARG_DEFINE_INTERFACE   ${ARG_DEFINE}        ${ARG_DEFINE_PRIVATE}  )
    list(APPEND ARG_C_OPTION_INTERFACE ${ARG_C_OPTION}      ${ARG_C_OPTION_PRIVATE})
    list(APPEND ARG_L_OPTION_INTERFACE ${ARG_L_OPTION}      ${ARG_L_OPTION_PRIVATE})
    list(APPEND ARG_PCH_INTERFACE      ${ARG_PCH_PUBLIC}    ${ARG_PCH}             )
    set(ARG_SOURCE_PUBLIC    "")
    set(ARG_SOURCE           "")
    set(ARG_INC              "")
    set(ARG_INC_PRIVATE      "")
    set(ARG_LIB              "")
    set(ARG_LIB_PRIVATE      "")
    set(ARG_DEFINE           "")
    set(ARG_DEFINE_PRIVATE   "")
    set(ARG_C_OPTION         "")
    set(ARG_C_OPTION_PRIVATE "")
    set(ARG_L_OPTION         "")
    set(ARG_L_OPTION_PRIVATE "")
    set(ARG_PCH_PUBLIC       "")
    set(ARG_PCH              "")

    if(NOT "${ARG_ADD_CURRENT_TO}" STREQUAL "NONE")
      set(ARG_ADD_CURRENT_TO "INTERFACE")
    endif()
  endif()
  
  # [option]
  # TEST
  # NOT_GROUP
  # [value]
  # MODE: EXE / STATIC / SHARED / INTERFACE / STATIC_AND_SHARED
  # ADD_CURRENT_TO: PUBLIC / INTERFACE / PRIVATE (default) / NONE
  # RET_TARGET_NAME
  # CXX_STANDARD: 11/14/17/20, default is global CXX_STANDARD (20)
  # PCH_REUSE_FROM
  # [list] : public, interface, private
  # SOURCE: dir(recursive), file, auto add currunt dir | target_sources
  # INC: dir                                           | target_include_directories
  # LIB: <lib-target>, *.lib                           | target_link_libraries
  # DEFINE: #define ...                                | target_compile_definitions
  # C_OPTION: compile options                          | target_compile_options
  # L_OPTION: link options                             | target_link_options
  # PCH: precompile headers                            | target_precompile_headers
  
  # sources
  if("${ARG_ADD_CURRENT_TO}" STREQUAL "PUBLIC")
    list(APPEND ARG_SOURCE_PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  elseif("${ARG_ADD_CURRENT_TO}" STREQUAL "INTERFACE")
    list(APPEND ARG_SOURCE_INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
  elseif("${ARG_ADD_CURRENT_TO}" STREQUAL "PRIVATE")
    list(APPEND ARG_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
  elseif(NOT "${ARG_ADD_CURRENT_TO}" STREQUAL "NONE")
    message(FATAL_ERROR "ADD_CURRENT_TO [${ARG_ADD_CURRENT_TO}] is not supported")
  endif()
  _Custom_ExpandSources(sources_public ARG_SOURCE_PUBLIC)
  _Custom_ExpandSources(sources_interface ARG_SOURCE_INTERFACE)
  _Custom_ExpandSources(sources_private ARG_SOURCE)
  
  # group
  if(NOT NOT_GROUP)
    set(allsources ${sources_public} ${sources_interface} ${sources_private})
    foreach(src ${allsources})
      get_filename_component(dir ${src} DIRECTORY)
      string(FIND ${dir} ${CMAKE_CURRENT_SOURCE_DIR} idx)
      if(NOT idx EQUAL -1)
        set(base_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
        file(RELATIVE_PATH rdir "${CMAKE_CURRENT_SOURCE_DIR}/.." ${dir})
      else()
        set(base_dir ${PROJECT_SOURCE_DIR})
      endif()
      file(RELATIVE_PATH rdir ${base_dir} ${dir})
      if(MSVC)
        string(REPLACE "/" "\\" rdir_MSVC ${rdir})
        set(rdir "${rdir_MSVC}")
      endif()
      source_group(${rdir} FILES ${src})
    endforeach()
  endif()
  
  # target folder
  file(RELATIVE_PATH targetRelPath "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(targetFolder "${PROJECT_NAME}/${targetRelPath}")
  
  Custom_GetTargetName(coreTargetName ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT "${ARG_RET_TARGET_NAME}" STREQUAL "")
    set(${ARG_RET_TARGET_NAME} ${coreTargetName} PARENT_SCOPE)
  endif()
  
  # print
  message(STATUS "- name: ${coreTargetName}")
  message(STATUS "- folder : ${targetFolder}")
  message(STATUS "- mode: ${ARG_MODE}")
  Custom_List_Print(STRS ${sources_private}
    TITLE  "- sources (private):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${sources_interface}
    TITLE  "- sources interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${sources_public}
    TITLE  "- sources public:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE}
    TITLE  "- define (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE_PRIVATE}
    TITLE  "- define interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE_INTERFACE}
    TITLE  "- define private:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_LIB}
    TITLE  "- lib (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_LIB_INTERFACE}
    TITLE  "- lib interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_LIB_PRIVATE}
    TITLE  "- lib private:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_INC}
    TITLE  "- inc (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_INC_INTERFACE}
    TITLE  "- inc interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_INC_PRIVATE}
    TITLE  "- inc private:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE}
    TITLE  "- define (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE_INTERFACE}
    TITLE  "- define interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_DEFINE_PRIVATE}
    TITLE  "- define private:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_C_OPTION}
    TITLE  "- compile option (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_C_OPTION_INTERFACE}
    TITLE  "- compile option interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_C_OPTION_PRIVATE}
    TITLE  "- compile option private:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_L_OPTION}
    TITLE  "- link option (public):"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_L_OPTION_INTERFACE}
    TITLE  "- link option interface:"
    PREFIX "  * ")
  Custom_List_Print(STRS ${ARG_L_OPTION_PRIVATE}
    TITLE  "- link option private:"
    PREFIX "  * ")
  
  set(targetNames "")

  # add target
  if("${ARG_MODE}" STREQUAL "EXE")
    add_executable(${coreTargetName})
    add_executable("Custom::${coreTargetName}" ALIAS ${coreTargetName})
    if(MSVC)
      set_target_properties(${coreTargetName} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${Custom_RootProjectPath}/bin")
    endif()
    set_target_properties(${coreTargetName} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
    set_target_properties(${coreTargetName} PROPERTIES MINSIZEREL_POSTFIX ${CMAKE_MINSIZEREL_POSTFIX})
    set_target_properties(${coreTargetName} PROPERTIES RELWITHDEBINFO_POSTFIX ${CMAKE_RELWITHDEBINFO_POSTFIX})
    list(APPEND targetNames ${coreTargetName})
  elseif("${ARG_MODE}" STREQUAL "STATIC")
    add_library(${coreTargetName} STATIC)
    add_library("Custom::${coreTargetName}" ALIAS ${coreTargetName})
    list(APPEND targetNames ${coreTargetName})
  elseif("${ARG_MODE}" STREQUAL "SHARED")
    add_library(${coreTargetName} SHARED)
    add_library("Custom::${coreTargetName}" ALIAS ${coreTargetName})
    target_compile_definitions(${coreTargetName} PRIVATE UCMAKE_EXPORT_${coreTargetName})
    list(APPEND targetNames ${coreTargetName})
  elseif("${ARG_MODE}" STREQUAL "INTERFACE")
    add_library(${coreTargetName} INTERFACE)
    add_library("Custom::${coreTargetName}" ALIAS ${coreTargetName})
    list(APPEND targetNames ${coreTargetName})
  elseif("${ARG_MODE}" STREQUAL "STATIC_AND_SHARED")
    add_library(${coreTargetName}_static STATIC)
    add_library("Custom::${coreTargetName}_static" ALIAS ${coreTargetName}_static)
    add_library(${coreTargetName}_shared SHARED)
    add_library("Custom::${coreTargetName}_shared" ALIAS ${coreTargetName}_shared)
    target_compile_definitions(${coreTargetName}_static PUBLIC UCMAKE_STATIC_${coreTargetName})
    target_compile_definitions(${coreTargetName}_shared PRIVATE UCMAKE_EXPORT_${coreTargetName})
    list(APPEND targetNames ${coreTargetName}_static ${coreTargetName}_shared)
  else()
    message(FATAL_ERROR "mode [${ARG_MODE}] is not supported")
    return()
  endif()

  foreach(targetName ${targetNames})
    if(NOT "${ARG_CXX_STANDARD}" STREQUAL "")
      set_property(TARGET ${targetName} PROPERTY CXX_STANDARD ${ARG_CXX_STANDARD})
      message(STATUS "- CXX_STANDARD : ${ARG_CXX_STANDARD}")
    endif()
  
    # folder
    if(NOT ${ARG_MODE} STREQUAL "INTERFACE")
      set_target_properties(${targetName} PROPERTIES FOLDER ${targetFolder})
    endif()
    
    # target sources
    foreach(src ${sources_public})
      get_filename_component(abs_src ${src} ABSOLUTE)
      file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
      target_sources(${targetName} PUBLIC
        $<BUILD_INTERFACE:${abs_src}>
        $<INSTALL_INTERFACE:${package_name}/${rel_src}>
      )
    endforeach()
    foreach(src ${sources_private})
      get_filename_component(abs_src ${src} ABSOLUTE)
      file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
      target_sources(${targetName} PRIVATE
        $<BUILD_INTERFACE:${abs_src}>
        $<INSTALL_INTERFACE:${package_name}/${rel_src}>
      )
    endforeach()
    foreach(src ${sources_interface})
      get_filename_component(abs_src ${src} ABSOLUTE)
      file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
      target_sources(${targetName} INTERFACE
        $<BUILD_INTERFACE:${abs_src}>
        $<INSTALL_INTERFACE:${package_name}/${rel_src}>
      )
    endforeach()
    
    # target define
    target_compile_definitions(${targetName}
      PUBLIC ${ARG_DEFINE}
      INTERFACE ${ARG_DEFINE_INTERFACE}
      PRIVATE ${ARG_DEFINE_PRIVATE}
    )
    
    # target lib
    target_link_libraries(${targetName}
      PUBLIC ${ARG_LIB}
      INTERFACE ${ARG_LIB_INTERFACE}
      PRIVATE ${ARG_LIB_PRIVATE}
    )
    
    # target inc
    foreach(inc ${ARG_INC})
      get_filename_component(abs_inc ${inc} ABSOLUTE)
      file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
      target_include_directories(${targetName} PUBLIC
        $<BUILD_INTERFACE:${abs_inc}>
        $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
      )
    endforeach()
    foreach(inc ${ARG_INC_PRIVATE})
      get_filename_component(abs_inc ${inc} ABSOLUTE)
      file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
      target_include_directories(${targetName} PRIVATE
        $<BUILD_INTERFACE:${abs_inc}>
        $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
      )
    endforeach()
    foreach(inc ${ARG_INC_INTERFACE})
      get_filename_component(abs_inc ${inc} ABSOLUTE)
      file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${inc})
      target_include_directories(${targetName} INTERFACE
        $<BUILD_INTERFACE:${abs_inc}>
        $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
      )
    endforeach()
    
    # target compile option
    target_compile_options(${targetName}
      PUBLIC ${ARG_C_OPTION}
      INTERFACE ${ARG_C_OPTION_INTERFACE}
      PRIVATE ${ARG_C_OPTION_PRIVATE}
    )
    
    # target link option
    target_link_options(${targetName}
      PUBLIC ${ARG_L_OPTION}
      INTERFACE ${ARG_L_OPTION_INTERFACE}
      PRIVATE ${ARG_L_OPTION_PRIVATE}
    )
    
    # target pch
    target_precompile_headers(${targetName}
      PUBLIC ${ARG_PCH_PUBLIC}
      INTERFACE ${ARG_PCH_INTERFACE}
      PRIVATE ${ARG_PCH}
    )

    # target pdb
    if (MSVC)
      set_target_properties(${targetName}
        PROPERTIES
        COMPILE_PDB_NAME_DEBUG ${targetName}${CMAKE_DEBUG_POSTFIX}
        COMPILE_PDB_NAME_RELEASE ${targetName}${CMAKE_RELEASE_POSTFIX}
        COMPILE_PDB_NAME_MINSIZEREL ${targetName}${CMAKE_MINSIZEREL_POSTFIX}
        COMPILE_PDB_NAME_RELWITHDEBINFO ${targetName}${CMAKE_RELWITHDEBINFO_POSTFIX})
    endif()
    
    if(NOT "${ARG_OUTPUT_NAME}" STREQUAL "")
      set_target_properties(${targetName} PROPERTIES OUTPUT_NAME "${ARG_OUTPUT_NAME}" CLEAN_DIRECT_OUTPUT 1)
    endif()
  
    if(NOT "${ARG_PCH_REUSE_FROM}" STREQUAL "")
      target_precompile_headers(${targetName} REUSE_FROM "${ARG_PCH_REUSE_FROM}")
    endif()
  
    if(NOT ARG_TEST)
      install(TARGETS ${targetName}
        EXPORT "${PROJECT_NAME}Targets"
        RUNTIME DESTINATION "bin"
        ARCHIVE DESTINATION "${package_name}/lib"
        LIBRARY DESTINATION "${package_name}/lib"
      )
      if("${ARG_MODE}" STREQUAL "STATIC" OR
        "${ARG_MODE}" STREQUAL "SHARED" OR
        "${ARG_MODE}" STREQUAL "STATIC_AND_SHARED")
        # dll
        install(FILES
          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}/${targetName}${CMAKE_DEBUG_POSTFIX}.pdb"
          CONFIGURATIONS Debug DESTINATION "bin" OPTIONAL
        )
        install(FILES
          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}/${targetName}${CMAKE_RELEASE_POSTFIX}.pdb"
          CONFIGURATIONS Release DESTINATION "bin" OPTIONAL
        )
        install(FILES
          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}/${targetName}${CMAKE_MINSIZEREL_POSTFIX}.pdb"
          CONFIGURATIONS MinSizeRel DESTINATION "bin" OPTIONAL
        )
        install(FILES
          "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO}/${targetName}${CMAKE_RELWITHDEBINFO_POSTFIX}.pdb"
          CONFIGURATIONS RelWithDebInfo DESTINATION "bin" OPTIONAL
        )
        # lib
        install(FILES
          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}/${targetName}${CMAKE_DEBUG_POSTFIX}.pdb"
          CONFIGURATIONS Debug DESTINATION "${package_name}/lib" OPTIONAL
        )
        install(FILES
          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}/${targetName}${CMAKE_RELEASE_POSTFIX}.pdb"
          CONFIGURATIONS Release DESTINATION "${package_name}/lib" OPTIONAL
        )
        install(FILES
          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}/${targetName}${CMAKE_MINSIZEREL_POSTFIX}.pdb"
          CONFIGURATIONS MinSizeRel DESTINATION "${package_name}/lib" OPTIONAL
        )
        install(FILES
          "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}/${targetName}${CMAKE_RELWITHDEBINFO_POSTFIX}.pdb"
          CONFIGURATIONS RelWithDebInfo DESTINATION "${package_name}/lib" OPTIONAL
        )
      endif()
    endif()
  endforeach()
  
  message(STATUS "----------")
endfunction()

macro(Custom_InitProject)
  set(CMAKE_DEBUG_POSTFIX "d")
  set(CMAKE_RELEASE_POSTFIX "")
  set(CMAKE_MINSIZEREL_POSTFIX "msr")
  set(CMAKE_RELWITHDEBINFO_POSTFIX "rd")
  
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED True)

  if(NOT CMAKE_BUILD_TYPE)
    message(NOTICE "No default CMAKE_BUILD_TYPE, so UCMake set it to \"Release\"")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  endif()

  add_compile_definitions(UCMAKE_CONFIG_$<UPPER_CASE:$<CONFIG>>)
  add_compile_definitions(
    $<$<CONFIG:Debug>:UCMAKE_CONFIG_POSTFIX="${CMAKE_DEBUG_POSTFIX}">
    $<$<CONFIG:Release>:UCMAKE_CONFIG_POSTFIX="">
    $<$<CONFIG:MinSizeRel>:UCMAKE_CONFIG_POSTFIX="${CMAKE_MINSIZEREL_POSTFIX}">
    $<$<CONFIG:RelWithDebInfo>:UCMAKE_CONFIG_POSTFIX="${CMAKE_RELWITHDEBINFO_POSTFIX}">
    $<$<NOT:$<OR:$<CONFIG:Debug>,$<CONFIG:Release>,$<CONFIG:MinSizeRel>,$<CONFIG:RelWithDebInfo>>>:UCMAKE_CONFIG_POSTFIX="">
  )
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # using Clang
    message(STATUS "Compiler: Clang ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
      message(FATAL_ERROR "clang (< 10) not support concept")
      return()
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(STATUS "Compiler: GCC ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
      message(FATAL_ERROR "gcc (< 10) not support concept")
      return()
    endif()
  # using GCC
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
    message(STATUS "Compiler: MSVC ${CMAKE_CXX_COMPILER_VERSION}")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "19.26")
      message(FATAL_ERROR "MSVC (< 1926 / 2019 16.6) not support concept")
      return()
    endif()
  else()
    message(WARNING "Unknown CMAKE_CXX_COMPILER_ID : ${CMAKE_CXX_COMPILER_ID}")
  endif()
  
  message(STATUS "CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
  
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    Custom_Path_Back(root ${CMAKE_INSTALL_PREFIX} 1)
    set(CMAKE_INSTALL_PREFIX "${root}/Custom" CACHE PATH "install prefix" FORCE)
  endif()
  
  if(NOT Custom_RootProjectPath)
    set(Custom_RootProjectPath ${PROJECT_SOURCE_DIR})
  endif()
  
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${Custom_RootProjectPath}/bin")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${Custom_RootProjectPath}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${Custom_RootProjectPath}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endmacro()
