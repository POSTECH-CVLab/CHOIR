# Self-Supervised Learning of Canonical Orientation

### Quick Start
Within a virtual environment (recommended), command as follows:
```bash
~$ conda create -n cores python=3.8 -y
~$ conda activate cores
(cores) ~$ pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
(cores) ~$ pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
(cores) ~$ pip install -r requirements-wo-pytorch3d.txt
(cores) ~$ bash install_torch_batch_svd.sh # optional (condor)
~$ apt-get update && apt-get install libgl1 # optional
```

All configurations for the experiment are described in `config.py`.


### Overview of the codebase

- Datasets
  - ShapeNet
    - [The preprocessed ShapeNetStabilityTest](https://postechackr-my.sharepoint.com/:u:/g/personal/p0125ch_postech_ac_kr/Eesr8Nh4RiFIl4rWPQgckaYBPJ9yGA4pAnNg2wtK31jVbw?e=uyO0Me)
  - ModelNet40 (todo)
  - ModelNet40-D (todo)

- SE(3)-equivariant backbones
  - Vector-Neuron with Translation (VNT)

- Heads
  - Global average pooling
  - Generalized mean pooling (GeM)
  - ...
  - Point Transformer + GeM + Concat-Fusion

- Metrics
  - Stability
  - Consistency
  - Average of the stability and consistency


### Misc
Use `vis.ipynb` and `debug.ipynb` for visualization and debugging, respectively.