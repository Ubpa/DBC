Ubpa_AddTarget(
  TEST
  MODE EXE
  CXX_STANDARD 17
  LIB
    Ubpa::CMakeLibTorch_NeuralSolver
    Ubpa::CMakeLibTorch_MoveDLLs
    Ubpa::CMakeLibTorch_bc7e
    "${PROJECT_SOURCE_DIR}/lib/nvtt30205.lib"
)
