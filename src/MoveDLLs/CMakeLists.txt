set(TargetName "UNKNOWN")
Ubpa_AddTarget(
  MODE STATIC
  RET_TARGET_NAME TargetName
  LIB
    "${TORCH_LIBRARIES}"
)
Ubpa_TorchCopyDll(${TargetName})
