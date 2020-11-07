# DeepLandscape: Adversarial Modeling of Landscape Videos
Implementation of DeepLandscape: Adversarial Modeling of Landscape Video in PyTorch
## [Project Page](https://saic-mdal.github.io/deep-landscape/) | [Video Explanation](https://youtu.be/mnYIx9DwVlE) | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680256.pdf) | [Teaser [1min]](https://youtu.be/2CoQRf5qXWY)
---

Official repository for the paper E. Logacheva, R. Suvorov, O. Khomenko, A. Mashikhin, and V. Lempitsky. "DeepLandscape: Adversarial Modeling of Landscape Videos" In 2020 European Conference on Computer Vision (ECCV).

![teaser image](./docs/img/01_intro_grid.jpg)

## GIF sample
<img src="./docs/img/gifs/2-1336_homman3.mp4_frames.gif" width="200" style="border-radius: 15px;" alt="gif sample #1">

---

## Requirements
* `pip3 install -r requirements.txt`

## Usage
Download approprate [everything from here](https://drive.google.com/drive/folders/1HqrT8SwkPOg_N9b2-KLZGUAi4OiLMFEz?usp=sharing) and put in the `results` directory.

### Homographies
Use ``homographies/manual_homographies`` to reproduce the paper; use ``homographies/manual_homographies_x2.5`` if you want the speed to match the speed of real videos in test data; use `homographies/selected_homographies` to get the best visual results.

### Animate Images
Move images you like to animate to `results/test_images`. Then run

``PYTHONPATH=`pwd`:$PYTHONPATH runfiles/encode_and_animate_test_all256_with_style.sh <homography_dir>``

Results will be saved in `results/encode_and_animate_results`.

### Generate Synthetic Videos
To use the 256x256 generator run

`./generate.py config/train/256.yaml --homography_dir <homography_dir>`

Results will be saved in `results/generated`

### Calculate Metrics
* Put test images into `results/test_images`.
* Split test videos to frames: `vid_dl/create_images.py <input directory> results/test_videos`.
* ``PYTHONPATH=`pwd`:$PYTHONPATH runfiles/calc_metrics_make_report_test_all256.sh <homography_dir>``

### Super-Resoultion
See [superres/README.md](superres/README.md)

### Train
#### Train the Generator
You should prepare an lmdb dataset:

`./prepare_data.py <data type (images or videos)> <input data path> --out <output lmdb directory> --n_worker <number of workers>`

To train the 256x256 generator:

`./train.py configs/train/256.yaml --restart -i <path to image data> -v <path to video data>`

#### Train the Encoder
You should prepare a dataset:

``PYTHONPATH=`pwd`:$PYTHONPATH runfiles/gen_encoder_train_data_256.sh``

To train the 256x256 encoder:

``PYTHONPATH=`pwd`:$PYTHONPATH runfiles/train_encoder_256.sh``

## Download Test Data
`vid_dl/main.py <output directory>`

-----
## Acknowledgment
This repository is based on [Kim Seonghyeon's Implementation A Style-Based Generator Architecture for Generative Adversarial Networks in PyTorch](https://github.com/rosinality/style-based-gan-pytorch)

The Superresolution part is based on [https://github.com/xinntao/BasicSR](https://github.com/xinntao/BasicSR)

Mean optical flow calculation is taken from https://github.com/avinashpaliwal/Super-SloMo

Segmentation is taken form https://github.com/CSAILVision/semantic-segmentation-pytorch

### Metrics
* LPIPS metric https://github.com/richzhang/PerceptualSimilarity
* SSIM https://github.com/Po-Hsun-Su/pytorch-ssim
* FID https://github.com/mseitzer/pytorch-fid

-----

## Citation
If you found our work useful, please don't forget to cite
```
@inproceedings{Logacheva_2020_ECCV,
  author = {Logacheva, Elizaveta and
            Suvorov, Roman and
            Khomenko, Oleg and
            Mashikhin, Anton and
            Lempitsky, Victor
  },
  title = {DeepLandscape: Adversarial Modeling of Landscape Videos},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year = {2020},
}
```
