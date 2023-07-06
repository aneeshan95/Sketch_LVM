# Sketch_LVM
### Official repository of _CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not_

## Accepted in **CVPR 2023**

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2303.13440.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-F9D371)](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Sain_CLIP_for_All_CVPR_2023_supplemental.pdf)
[![video](https://img.shields.io/badge/Video-Presentation-B85252)](https://www.youtube.com/watch?v=ImcQFsS1SfE)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://aneeshan95.github.io/Sketch_LVM/)

## Abstract
 
![teaser](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/opener.png?raw=true)
 
In this paper, we leverage CLIP for zero-shot sketch based image retrieval (ZS-SBIR). We are largely inspired by recent advances on foundation models and the unparalleled generalisation ability they seem to offer, but for the first time tailor it to benefit the sketch community. We put forward novel designs on how best to achieve this synergy, for both the category setting and the fine-grained setting ("all"). At the very core of our solution is a prompt learning setup. First we show just via factoring in sketch-specific prompts, we already have a category-level ZS-SBIR system that overshoots all prior arts, by a large margin (24.8%) - a great testimony on studying the CLIP and ZS-SBIR synergy. Moving onto the fine-grained setup is however trickier, and requires a deeper dive into this synergy. For that, we come up with two specific designs to tackle the fine-grained matching nature of the problem: (i) an additional regularisation loss to ensure the relative separation between sketches and photos is uniform across categories, which is not the case for the gold standard standalone triplet loss, and (ii) a clever patch shuffling technique to help establishing instance-level structural correspondences between sketch-photo pairs. With these designs, we again observe significant performance gains in the region of 26.9% over previous state-of-the-art. The take-home message, if any, is the proposed CLIP and prompt learning paradigm carries great promise in tackling other sketch-related tasks (not limited to ZS-SBIR) where data scarcity remains a great challenge.

## Architecture

Cross-category FG-ZS-SBIR. A common (photo-sketch) learnable visual prompt shared across categories is trained using CLIP’s image encoder over three losses as shown. CLIP’s text-encoder based classification loss is used during training.

![arch](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/arch.png?raw=true)

 ## Datasets
For ZS-SBIR we used the [Sketchy](https://github.com/AnjanDutta/sem-pcyc/) (extended), [TUBerlin](https://github.com/AnjanDutta/sem-pcyc/) and [QuickDraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) (a smaller version).
For Fine-grained ZS-SBIR we used the [Sketchy](https://github.com/AnjanDutta/sem-pcyc/) (basic) dataset with fine-grained sketch-photo associations.

 
## Qualitative Results

Qualitative results of ZS-SBIR on Sketchy by a baseline (blue) method vs Ours (green).
![qualitative_category](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/qual_cat.png?raw=true)


Qualitative results of FG-ZS-SBIR on Sketchy by a baseline (blue) method vs Ours (green). The images are arranged in increasing order of the ranks beside their corresponding sketch-query, i.e the left-most image was retrieved at rank-1 for every category. The true-match for every query, if appearing in top-5 is marked in a green frame. Numbers denote the rank at which that true-match is retrieved for every corresponding sketch-query.
![qualitative_FG](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/qual_FG.png?raw=true)


## Quantitative Results

Quantitative results of our method against a few SOTAs.
![qualitative_FG](https://github.com/aneeshan95/Sketch_LVM/blob/main/static/images/quant.png?raw=true)


 ## Code
 
 A workable basic version of the code for CLIP adapted for ZS-SBIR has been uploaded.
 - `src` folder holds the source files.
 - `experiments` folder holds the executable wrapper for the model with particular specifications.

An example command to run the code is given below:
```shell
$ cd Sketch_LVM
$ python -m experiments.LN_prompt --exp_name=clip_split5 --n_prompts=3 --clip_lr=1e-6 --data_split=0.5
```

The code for cross-category Fine-Grained ZS-SBIR will be uploaded in some time.

## Bibtex

Please cite our work if you found it useful. Thanks.
```
@Inproceedings{sain2023clip,
  title={{CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not}},
  author={Aneeshan Sain and Ayan Kumar Bhunia and Pinaki Nath Chowdhury and Subhadeep Koley and Tao Xiang and Yi-Zhe Song},
  booktitle={CVPR},
  year={2023}
}
```
