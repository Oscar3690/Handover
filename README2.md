# JupyterLab ImageNet Setup

These notes are for running `examples/models/imagenet_example.ipynb` in
JupyterLab.

## 1. Put the ImageNet validation tar in `datasets`

The large ImageNet validation archive is not included in this repository.
Place it here:

```text
datasets/ILSVRC2012_img_val.tar
```

In the Docker/JupyterLab environment used by this project, that usually means:

```text
/home/jovyan/transaxx/datasets/ILSVRC2012_img_val.tar
```

Do not manually extract this tar. Use the preparation script below so the
validation images are sorted into the folder format expected by the notebook.

The small devkit directory is already included in this repository:

```text
datasets/ILSVRC2012_devkit_t12/
```

You do not need to upload or extract `ILSVRC2012_devkit_t12.tar.gz`.

## 2. Create `imagenet_data1`

Open a JupyterLab terminal and run:

```bash
cd /home/jovyan/transaxx
python datasets/prepare_imagenet_data1.py
```

This reads:

```text
datasets/ILSVRC2012_img_val.tar
datasets/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
datasets/ILSVRC2012_devkit_t12/data/meta.mat
```

and creates:

```text
datasets/imagenet_data1/
  val/
  train_tiny/
```

`val` contains the full validation set in `ImageFolder` format.
`train_tiny` is a small subset copied or linked from validation. By default it
uses 5 images per class.

Useful options:

```bash
python datasets/prepare_imagenet_data1.py --tiny-per-class 10 --overwrite
python datasets/prepare_imagenet_data1.py --copy-tiny
```

Use `--overwrite` only when you intentionally want to rebuild
`datasets/imagenet_data1`.

## 3. Update notebook paths if your repo is not in `/home/jovyan/transaxx`

In `examples/models/imagenet_example.ipynb`, the first setup cell now tries to
find the repository root automatically. The repository root is the folder that
directly contains:

```text
classification/
pytorch-quantization/
datasets/
examples/
```

If the notebook cannot find that folder, set the `TRANSAXX_PATH` environment
variable before launching JupyterLab, or edit the first setup cell manually.

```python
TRANSAXX_PATH = "/home/jovyan/transaxx"
IMAGENET_PATH = "/home/jovyan/transaxx/datasets/imagenet_data1"
VAL_DIR = os.path.join(IMAGENET_PATH, "val")
CALIB_DIR = os.path.join(IMAGENET_PATH, "train_tiny")
DEVKIT_ROOT = "/home/jovyan/transaxx/datasets"
```

If you see `ModuleNotFoundError: No module named 'classification'`, then
`TRANSAXX_PATH` is pointing at the wrong folder. It must point to the repo root,
not to `examples`, `examples/models`, or `datasets`.

For example, if the repo is mounted at `/workspace/Handover`, use:

```python
TRANSAXX_PATH = "/workspace/Handover"
IMAGENET_PATH = "/workspace/Handover/datasets/imagenet_data1"
DEVKIT_ROOT = "/workspace/Handover/datasets"
```

If you run JupyterLab directly on Windows, use raw strings:

```python
TRANSAXX_PATH = r"C:\Users\umida\Desktop\Handover"
IMAGENET_PATH = r"C:\Users\umida\Desktop\Handover\datasets\imagenet_data1"
DEVKIT_ROOT = r"C:\Users\umida\Desktop\Handover\datasets"
```

## 4. Run the notebook

After `imagenet_data1` exists, open:

```text
examples/models/imagenet_example.ipynb
```

Run the first code cell once to install `requirements.txt` into the active
Jupyter kernel. This avoids terminal installs for packages such as `timm`.

Run the setup cells first. The notebook should print paths similar to:

```text
VAL_DIR: /home/jovyan/transaxx/datasets/imagenet_data1/val
CALIB_DIR: /home/jovyan/transaxx/datasets/imagenet_data1/train_tiny
```

The preparation script creates numeric ImageNet devkit ID folders by default
(`1` through `1000`), which matches the label-fix workflow already in the
notebook.

## 5. LUT files

The approximate multiplier headers are in:

```text
ext_modules/include/nn/cuda/axx_mults/
```

The available FPGA ISH headers include:

```text
mul8s_FPGA_ISH1.h
mul8s_FPGA_ISH2.h
mul8s_FPGA_ISH3.h
mul8s_FPGA_ISH4.h
mul8s_FPGA_ISH5.h
```

## 6. Test a different ImageNet dataset

The notebook uses this layout:

```text
<IMAGENET_PATH>/
  val/
  train_tiny/
```

Both `val` and `train_tiny` must be readable by `torchvision.datasets.ImageFolder`,
meaning each class has its own subfolder.

To prepare a different validation tar, run the script with explicit input and
output paths:

```bash
python datasets/prepare_imagenet_data1.py \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --output-dir /path/to/my_imagenet_data \
  --tiny-per-class 10
```

Then in the first notebook setup cell, point `IMAGENET_PATH` to that output:

```python
IMAGENET_PATH = "/path/to/my_imagenet_data"
VAL_DIR = os.path.join(IMAGENET_PATH, "val")
CALIB_DIR = os.path.join(IMAGENET_PATH, "train_tiny")
```

If your dataset is already staged in ImageFolder format, skip the preparation
script and only change `IMAGENET_PATH`. For supervised evaluation, the folder
names/order must still match the label mapping used by the notebook. The default
preparation script creates numeric ImageNet devkit ID folders (`1` through
`1000`), which is what the canonical label-fix cells expect.

For calibration-only runs, `train_tiny` labels are less important, but the folder
structure must still be valid for `ImageFolder`.

## 7. Change the approximate multiplier in the notebook

In `examples/models/imagenet_example.ipynb`, go to the section named
`7. Run approximate evaluation`.

Find this line:

```python
axx_list[0:47] = [{'axx_mult' : 'mul8s_1L2H', 'axx_power' : 0.8871, 'quant_bits' : 8, 'fake_quant' : False}] * 47
```

Change `axx_mult` to the header name without `.h`. For example:

```python
axx_list[0:47] = [{'axx_mult' : 'mul8s_FPGA_ISH3', 'axx_power' : 1.0, 'quant_bits' : 8, 'fake_quant' : False}] * 47
```

Use one of:

```text
mul8s_FPGA_ISH1
mul8s_FPGA_ISH2
mul8s_FPGA_ISH3
mul8s_FPGA_ISH4
mul8s_FPGA_ISH5
```

Also update `axx_power` if you want the reported power estimate to match the
selected multiplier. The notebook currently uses the same range syntax,
`axx_list[0:47]`, to approximate the first 47 linear layers; change the range
if you want to approximate fewer or more layers.

If a notebook or script contains a hard-coded path such as
`/home/jovyan/transaxx/...`, change it to the actual repository path in your
JupyterLab environment.
