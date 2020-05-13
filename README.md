# CenterUnet
CenterNet Unet Version.
Base on https://github.com/xingyizhou/CenterNet.
But remove focal loss convolution magic number and sigmoid clamp trick. Easier to do training and debuging. Flexible to replace backbone due to using qubel's segmentation models framework.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
    
## Major Changes
<pre>
1. Turn ground truth to binary values for focal loss. Very effective for dealing with positive and negtive samples inbalance issue. Experiments show that Gaussian Distribution center point still can be learned after this change.
2. Add draw_elipse_gaussian for adaptive width/height.
3. Support unet framework, easy to add fpn and linknet which are supported by qubvel's framework.
4. One decoder for center point, one decoder for box and offset prediction. These two decoders share one encoder.
5. Output instance mask using InstanceFCN method.
6. Add instance box area size level as label. Total 6 classes, but every size has two adjacent size level labels.
</pre>

## Install
<pre>
python3 -m pip install torch==1.4.0
python3 -m pip install torchvision==0.4.2
python3 -m pip install pretrainedmodels==0.7.4
python3 -m pip install opencv-python
python3 -m pip install numpy
python3 -m pip install pytorch
python3 -m pip install pycocotools
python3 -m pip install cython
python3 -m pip install matplotlib
python3 -m pip install progress
python3 -m pip install numba
cd external; make
cd models/py_utils/_cpools/ python setup.py install --user
</pre>

## Prepare data
<pre>
1. Download coco 2017 dataset, unzip to your local directory.
2. mkdir data and setup symlink 'coco' to it.
3. Or just unzip it under data directory, depends on your choice. Like below:

(base) ubuntu@ubuntu:~/Training/CenterUnet/data/coco$ tree -L 1
.
├── annotations
├── test2017
├── train2017
├── val2017
</pre>

## Training
> Example
<pre>
Center phase train:
  python3 -u main.py --network_type unetobj --backbone resnet34 --batch_size 10 --train_phase pre_train_center --lr 0.001
Box phase train:
  python3 -u main.py --network_type unetobj --backbone resnet34 --batch_size 10 --train_phase pre_train_box --lr 0.0001 --resume
</pre>

## Check Result
<pre>
Check ./results for temporary training results. Try ./http.sh and open http://yourip:8000
Example for inferencing one image: python3 testimg.py --without_gpu --network_type unetobj --backbone resnet34 --nms --center_thresh 0.25 --image ./testsamples/horse.jpg
</pre>

## Most code from:
<pre>
https://github.com/xingyizhou/CenterNet
https://github.com/Duankaiwen/CenterNet
https://github.com/qubvel/segmentation_models.pytorch
https://github.com/yxlijun/Pelee.Pytorch
</pre>

![avatar](https://github.com/xuduo35/CenterUnet/blob/master/samples/centernet1.jpg?raw=true)

![avatar](https://github.com/xuduo35/CenterUnet/blob/master/samples/centernet2.jpg?raw=true)

![avatar](https://github.com/xuduo35/CenterUnet/blob/master/samples/centernet3.jpg?raw=true)

## License

CenterUnet itself is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from other open source projects mentioned above(section 'Most code from'). Please refer to the original License of these projects.
