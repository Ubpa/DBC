set(TargetName "UNKNOWN")
Ubpa_AddTarget(
  TEST
  MODE EXE
  CXX_STANDARD 17
  RET_TARGET_NAME TargetName
  LIB
    "${TORCH_LIBRARIES}"
)

Ubpa_TorchCopyDll(${TargetName})
