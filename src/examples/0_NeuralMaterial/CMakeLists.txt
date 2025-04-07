Ubpa_AddTarget(
  TEST
  MODE EXE
  CXX_STANDARD 17
  LIB
    Ubpa::DBC_DBC
    Ubpa::DBC_MoveDLLs
    "${PROJECT_SOURCE_DIR}/lib/nvtt30205.lib"
)
