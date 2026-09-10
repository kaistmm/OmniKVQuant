# Qwen2.5-Omni-3B checkpoint

`v_rotation.pt` is loaded automatically when running `--method omnikvquant`.

To generate another checkpoint, follow the [calibration commands](../README.md).
Select the generated file during inference with `--rotations /path/to/v_rotation.pt`.
