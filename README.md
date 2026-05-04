### `CHOIR`: `CH`aracteristic `O`rientation with `I`nvariant `R`esidual Learning

> [Stable and Consistent Prediction of 3D Characteristic Orientation via Invariant Residual Learning](https://arxiv.org/abs/2306.07547)\
> [Seungwook Kim<sup>1*</sup>](https://wookiekim.github.io/),
> [Chunghyun Park<sup>1*</sup>](https://chrockey.github.io/),
> [Jaesik Park<sup>2</sup>](http://jaesik.info/), and
> [Minsu Cho<sup>1</sup>](http://cvlab.postech.ac.kr/~mcho/) (*equal contribution)<br>
> <sup>1</sup>POSTECH and <sup>2</sup>Seoul National University<br>
> ICML 2023, Honolulu.

<div align="left">
  <a href="https://arxiv.org/abs/2306.07547"><img src="https://img.shields.io/badge/arXiv-2306.07547-b31b1b.svg"/></a>
</div>

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

> [!NOTE]
> CUDA is required for `torch-cluster` (compiled from source during `uv sync`). Tested on Python 3.12, PyTorch 2.11, CUDA 12.8, 8 NVIDIA A6000 GPUs.

## Data

ShapeNet point clouds are automatically downloaded and prepared on first run. To prepare manually:

```bash
uv run python src/data/prepare.py
```

This downloads ShapeNet from [AtlasNetV2](https://github.com/TheoDEPRELLE/AtlasNetV2), converts PLY to H5, and generates the stability evaluation set.

## Training

```bash
# Single-class (8 GPU DDP)
uv run python src/train.py experiment=airplane
uv run python src/train.py experiment=car
uv run python src/train.py experiment=chair
uv run python src/train.py experiment=table

# Multi-class (airplane, car, chair, table)
uv run python src/train.py experiment=multi

# Override any config
uv run python src/train.py experiment=airplane trainer.devices=1 data.batch_size=16
```

Checkpoints and logs are saved to `logs/`.

## Evaluation

```bash
uv run python src/eval.py ckpt_path=logs/.../best.ckpt
```

Metrics reported: **consistency** (cross-instance angular std) and **stability** (within-instance angular std), per class and averaged.

## Acknowledgments

This project builds upon [Vector Neurons](https://github.com/FlyingGiraworffe/vnn) for SO(3)-equivariant layers and [Canonical Capsules](https://github.com/canonical-capsules/canonical-capsules) for ShapeNet data processing. The codebase structure follows [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{kim2023choir,
  title={Stable and Consistent Prediction of 3D Characteristic Orientation via Invariant Residual Learning},
  author={Kim, Seungwook and Park, Chunghyun and Park, Jaesik and Cho, Minsu},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2023}
}
```
