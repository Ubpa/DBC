Custom_AddTarget(
  TEST
  MODE EXE
  CXX_STANDARD 17
  LIB
    Custom::DBC_DBC
    Custom::DBC_MoveDLLs
    "${PROJECT_SOURCE_DIR}/lib/nvtt30205.lib"
)
