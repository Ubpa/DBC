set(mode "")
set(private_defines "")
if(Ubpa_UMBD_UsingSharedLibrary)
  set(mode SHARED)
  list(APPEND private_defines "UBPA_UMBD_BUILDING_SHARED_LIBRARY")
else()
  set(mode STATIC)
endif()

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/bin/bc7e.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_avx.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_sse2.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_sse4.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_avx2.obj
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/ispc.exe
	  -g -O2 ${CMAKE_CURRENT_SOURCE_DIR}/bc7e.ispc -o ${CMAKE_SOURCE_DIR}/bin/bc7e.obj
	  -h bc7e_ispc.h --target=sse2,sse4,avx,avx2 --opt=fast-math --opt=disable-assertions
	DEPENDS bc7e.ispc)

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
  INC
    "${PROJECT_SOURCE_DIR}/include/bc7e"
  SOURCE
    "${PROJECT_SOURCE_DIR}/include/bc7e"
    "bc7e.ispc"
	${CMAKE_SOURCE_DIR}/bin/bc7e.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_avx.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_avx2.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_sse2.obj ${CMAKE_SOURCE_DIR}/bin/bc7e_sse4.obj
  DEFINE
	"SUPPORT_BC7E=1"
  C_OPTION_PRIVATE
    ${c_options_private}
)
