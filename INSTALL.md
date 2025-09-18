## Installation

### Requirements
- Linux or macOS with Python ≥ 3.10
- PyTorch ≥ 1.7 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Diffusers: install by source
- OpenCV is optional but needed by demo and visualization
- `pip install -r requirements.txt`

An example of installation is shown below:

```
git clone https://github.com/kchyun/Seg4Diff.git
cd Seg4Diff
conda create -n seg4diff python=3.10
conda activate seg4diff
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# install diffusers by source
cd diffusers
pip install -e .

pip install -r requirements.txt
```