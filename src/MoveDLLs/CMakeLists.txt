set(TargetName "UNKNOWN")
Custom_AddTarget(
  MODE STATIC
  RET_TARGET_NAME TargetName
  LIB
    "${TORCH_LIBRARIES}"
)
Custom_TorchCopyDll(${TargetName})
