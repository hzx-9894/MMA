# MA:Mixed Attention Backbone towards 3D Weakly-supervised Detection  

## Preparation

Our code can run in the following environments:

``` Python 3.6.13

PyTorch 1.4.0

numpy 1.19.2

open3d 0.9.0.0

opencv-python 4.6.0.66

plyfile 0.7.4

```

The [Matterport3D](https://niessner.github.io/Matterport/) and [ScanNet](http://www.scan-net.org/) datasets can be downloaded from their official websites and processed using [BR](https://github.com/xuxw98/BackToReality)'s [data generation](https://github.com/xuxw98/BackToReality/tree/main/data_generation). Alternatively, you can also download the preprocessed datasets provided by BR directly:[ScanNet](https://drive.google.com/drive/folders/1hKjYXdHIpk8a1IPG_k4WmFacSPlfTYwZ),[Matterport](https://drive.google.com/drive/folders/166w4w9xa8c7WITDAGswEsDQ8cJxBJomn).

The final data organization shoule be:
* Mixed-Attention
    * ...
    * matterport
        * ...
        * matterport_train_detection_data_md40
        * matterport_train_detection_data_md40_obj_aug
        * matterport_train_detection_data_md40_obj_mesh_aug
    * scannet
        * ...
        * scannet_train_detection_data_md40
        * scannet_train_detection_data_md40_obj_aug
        * scannet_train_detection_data_md40_obj_mesh_aug


## Run

### GF3D-MA for Fully-supervised detection

```

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 1234 --nproc_per_node 4 train_GF_FSB_MA.py --num_point 50000 --num_decoder_layers 4 --size_delta 0.111111111111 --center_delta 0.04 --learning_rate 0.004 --decoder_learning_rate 0.0004 --weight_decay 0.0005 --dataset scannet --log_dir log_FSB --max_epoch 200 --batch_size 4


```

### BRM-MA

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 1234 --nproc_per_node 3 train_GF_BR_MA.py --num_point 50000 --num_decoder_layers 2 --size_delta 0.111111111111 --center_delta 0.04 --learning_rate 0.004 --decoder_learning_rate 0.0004 --weight_decay 0.0005 --dataset scannet --log_dir log_BR --optimizer adan --max_epoch 300 --batch_size 4

```

### GF3D-MA for Weakly-supervised detection

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 1234 --nproc_per_node 2 train_GF_WSB_MA.py --num_point 50000 --num_decoder_layers 2 --size_delta 0.111111111111 --center_delta 0.04 --learning_rate 0.004 --decoder_learning_rate 0.0004 --weight_decay 0.0005 --dataset scannet --log_dir log_WSB --max_epoch 300 --batch_size 4

```

Change the dataset by replacing the `--dataset scannet` with `--dataset matterport`.

In addition,if you want to test the detector 
original PointNet++,you can replace the [backbone_module.py](https://github.com/hzx-9894/Mixed-Attention/blob/main/models/backbone_module.py) with the  [backbone_module_org.py](https://github.com/hzx-9894/Mixed-Attention/blob/main/models/backbone_module_org.py).

## Acknowledgements

We thank the effort of [BR](https://github.com/xuxw98/BackToReality).
