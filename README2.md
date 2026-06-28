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

The small devkit directory should already exist:

```text
datasets/ILSVRC2012_devkit_t12/
```

If it is missing and you have `ILSVRC2012_devkit_t12.tar.gz`, put that archive
in `datasets` and extract it from a JupyterLab terminal:

```bash
cd /home/jovyan/transaxx/datasets
tar -xzf ILSVRC2012_devkit_t12.tar.gz
```

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

In `examples/models/imagenet_example.ipynb`, edit the first setup cell if your
JupyterLab path is different:

```python
TRANSAXX_PATH = "/home/jovyan/transaxx"
IMAGENET_PATH = "/home/jovyan/transaxx/datasets/imagenet_data1"
VAL_DIR = os.path.join(IMAGENET_PATH, "val")
CALIB_DIR = os.path.join(IMAGENET_PATH, "train_tiny")
DEVKIT_ROOT = "/home/jovyan/transaxx/datasets"
```

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

If a notebook or script contains a hard-coded path such as
`/home/jovyan/transaxx/...`, change it to the actual repository path in your
JupyterLab environment.
