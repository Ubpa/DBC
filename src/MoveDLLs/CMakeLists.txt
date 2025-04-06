set(TargetName "UNKNOWN")
Ubpa_AddTarget(
  MODE STATIC
  RET_TARGET_NAME TargetName
  LIB
    "${TORCH_LIBRARIES}"
    Ubpa::CMakeLibTorch_MoveDLLs
)
Ubpa_TorchCopyDll(${TargetName})
