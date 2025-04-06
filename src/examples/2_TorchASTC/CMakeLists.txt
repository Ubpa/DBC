set(TargetName "UNKNOWN")

Ubpa_AddTarget(
  TEST
  MODE EXE
  CXX_STANDARD 17
  RET_TARGET_NAME TargetName
  SOURCE
    "${PROJECT_SOURCE_DIR}/include/NeuralAidedMBD"
  INC
    "${PROJECT_SOURCE_DIR}/include/NeuralAidedMBD"
  LIB
    "${TORCH_LIBRARIES}"
    Ubpa::CMakeLibTorch_MoveDLLs
)
