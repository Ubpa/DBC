# Differentiable Block Compression for Neural Texture

EGSR 2025

36th Eurographics Symposium on Rendering

PDF: [Eurographics Digital Library](https://diglib.eg.org/items/9dfee807-eeec-485a-82a6-f466268d28c2)

Authors

- Tao Zhuang: Tencent Technology (Shenzhen) Co., LTD
- Wentao Liu: University of Science and Technology
- Ligang Liu: University of Science and Technology

Citation
```
@inproceedings{10.2312:sr.20251199,
booktitle = {Eurographics Symposium on Rendering},
editor = {Wang, Beibei and Wilkie, Alexander},
title = {{Differentiable Block Compression for Neural Texture}},
author = {Zhuang, Tao and Liu, Wentao and Liu, Ligang},
year = {2025},
publisher = {The Eurographics Association},
ISSN = {1727-3463},
ISBN = {978-3-03868-292-9},
DOI = {10.2312/sr.20251199}
}
```

## Environment

- CUDA 12.4.1

  > Additionally, it is necessary to install Nsight NVTX from CUDA 11.8.
  > Reference: Reply from user Te93 in [Failed to find nvToolsExt - C++ - PyTorch Forums](https://discuss.pytorch.org/t/failed-to-find-nvtoolsext/179635/2)

- LibTorch 2.4.1

- VS 2022 latest

- CMake latest

## Compilation

Use cmake to generate the sulotion, build the project `DBC_examples_NeuralMterial`

## Run

run the built exe (bin/DBC_examples_NeuralMaterial.exe) with command arguments

```
<object_name> <codec> <optimize_mode> <MoP> <Ns> <Nr> <feature_size>
```

for example

```
lubricant_spray BC7 1 1 2 2 512
```

It will compress `bin/image/lubricant_spray` texture set on the Neural Material model with DBC framework.

For more detailed information, please refer to the source code.

## Result

(decompressed) result images is located in `bin/image/pred`
