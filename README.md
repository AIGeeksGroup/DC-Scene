# <img src="https://github.com/AIGeeksGroup/DC-Scene/blob/main/images/dc-scene-logo.png" alt="logo" width="50"/> DC-Scene

This is the code repository for the paper:
> **DC-Scene: Data-Centric Learning for 3D Scene Understanding**
>
> [Ting Huang](https://github.com/Believeht029)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, [Ruicheng Zhang](https://github.com/SYSUzrc)\* and [Yang Zhao](https://scholar.google.com.au/citations?user=UrVEK7IAAAAJ&hl=en&oi=sra)\**
>
> \*Equal contribution. †Project lead. \**Corresponding author
>
>
> **[[arXiv]](https://arxiv.org/abs/2505.15232)** **[[Paper with Code]](https://paperswithcode.com/paper/dc-scene-data-centric-learning-for-3d-scene)**
> 
<center class='img'>
<img title="Framework of DC-Scene." src="https://github.com/AIGeeksGroup/DC-Scene/blob/main/images/image.png" width="100%">
</center>

## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
@article{dcscene2025,
  title={DC-Scene: Data-Centric Learning for 3D Scene Understanding},
  author={Huang, Ting and Zhang, Zeyu and Zhang, Ruicheng and Zhao, Yang},
  journal={arXiv preprint arXiv:2505.15232},
  year={2025}
}
```

## Introduction
3D scene understanding plays a fundamental role in vision applications such as robotics, autonomous driving, and augmented reality. However, advancing learning-based 3D scene understanding remains challenging due to two key limitations: (1) the large scale and complexity of 3D scenes lead to higher computational costs and slower training compared to 2D counterparts; and (2) high-quality annotated 3D datasets are significantly scarcer than those available for 2D vision. These challenges underscore the need for more efficient learning paradigms. In this work, we propose **DC-Scene**, a data-centric framework tailored for 3D scene understanding, which emphasizes enhancing data quality and training efficiency. Specifically, we introduce a CLIP-driven dual-indicator quality (DIQ) filter, combining vision-language alignment scores with caption-loss perplexity, along with a curriculum scheduler that progressively expands the training pool from the top 25\% to 75\% of scene–caption pairs. This strategy filters out noisy samples and significantly reduces dependence on large-scale labeled 3D data. Extensive experiments on ScanRefer and Nr3D demonstrate that DC-Scene achieves state-of-the-art performance (**86.1 CIDEr with the top-75\% subset vs. 85.4 with the full dataset**) while reducing training cost by approximately two-thirds, confirming that a compact set of high-quality samples can outperform exhaustive training.

<!-- ![image](https://github.com/AIGeeksGroup/DC-Scene/blob/main/image.png)-->
## Core Features
- **CLIP-driven dual-indicator quality (DIQ) filter**: Combines visual-language alignment score and description loss perplexity to effectively identify and filter low-quality scene-description pairs.
- **Curriculum learning scheduler**: Adopts a strategy of gradually expanding the training set to achieve a training process from easy to difficult.


## Requirements
- Python >= 3.7
- PyTorch >= 1.8
- CUDA-compatible GPU

## Environment Setup
You can set up your own conda virtual environment by running the commands below.

```bash
# create a clean conda environment from scratch
conda create --name dcscene python=3.8
conda activate dcscene
# install required packages
pip install -r requirements.txt
```

## Training
### Import Dataset Path
There are two datasets that need to set paths. The Scanrefer dataset sets the `DATASET_ROOT_DIR` and `DATASET_METADATA_DIR` global variables in the `datasets/scene_scanrefer.py` file. The Nr3D dataset also sets two global variables in the `datasets/scene_nr3d.py` file.

Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start Training
```bash
# w/o 2D input
python main.py --use_color --use_normal --checkpoint_dir ckpt/DC_Scene
# w/ 2D input
python main.py --use_color --use_multiview --checkpoint_dir ckpt_2D/DC_Scene
```

## Evaluation

There are two datasets that need to set paths. The Scanrefer dataset sets the `DATASET_ROOT_DIR` and `DATASET_METADATA_DIR` global variables in the `datasets/scene_scanrefer.py` file. The Nr3D dataset also sets two global variables in the `datasets/scene_nr3d.py` file.

Please modify the paths to match your actual dataset paths, set the training parameters, and then start model training.

### Start testing
```bash
# w/o 2D input
python main.py --use_color --use_normal --test_ckpt ckpt/DC_Scene/checkpoint_best.pth --test_caption
# w/ 2D input
python main.py --use_color --use_multiview --test_ckpt ckpt_2D/DC_Scene/checkpoint_best.pth --test_caption
```

