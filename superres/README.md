# How to run
Original code is taken from [https://github.com/xinntao/BasicSR](https://github.com/xinntao/BasicSR)

1. Download (`decoder_epoch_20.pth` and `encoder_epoch_20.pth`) for segmentation ade20k from [here](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/) and put them into `weights/ade20k-resnet50dilated-ppm_deepsup` folder

2. Download pretrained weight (`G.pth`) for our model from [drive](https://drive.google.com/file/d/1bDQ3MBNplWMyb5E5K7E5ThudRYcWRWjr/view?usp=sharing) and put it into `weights` folder

3. Run `CUDA_VISIBLE_DEVICES=0 python3 src/predict.py`

4. All results are stored in `outputs`