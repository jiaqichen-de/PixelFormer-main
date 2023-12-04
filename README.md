## PixelFormerPlus and SemanticBins
This is the official PyTorch implementation of model PixelFormerPlus and SemanticBins  
The optimization for WACV 2023 paper 'Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention'.

**[Paper](https://arxiv.org/pdf/2210.09071)** <br />
## Contents
1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Testing](#testing)
6. [Pretrained Models](#pretrainedmodels)
7. [Models](#models)


## Installation
Os ubuntu18.04
```
conda create -n pixelformer python=3.8
conda activate pixelformer
conda install pytorch=1.10.0 torchvision cudatoolkit=11.1
pip install matplotlib tqdm tensorboardX timm mmcv eiops
```
## Datasets
You can download the annotated depth maps data set of KITTI from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and download the raw data of KITTI form [here](https://www.cvlibs.net/datasets/kitti/raw_data.php)  
You can download the annotated depth maps data set of NYUv2 from [here](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
the raw data of NYUv2 from [here](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/newcrfs/datasets/nyu/sync.zip)   
The original dataset and annotated data downloaded from the official website are provided in the .mat file format. Therefore, you need to convert the .mat files into the .png format that can be recognized as input. Please refer to the open source code from [here](https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset)  
You can download the annotated depth maps data set and the raw data of HabitatDyn from [here](https://drive.google.com/drive/folders/1evlVaoB-EO3mNX15dgDNhluivg6KHyGr) 
 
## Training
First download the pretrained encoder backbone Swin-L 22K model from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the PixelFormer model on NYUv2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
```
python PixelFormerPlus/train.py arguments_train_nyu.txt
```

Training the PixelFormer model on KITTI:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
```
python PixelFormerPlus/train.py arguments_train_kittieigen.txt
```
Training the PixelFormerPlus model on NYUv2:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files
```
python PixelFormerPlus/train.py arguments_train_nyu.txt
```

Training the PixelFormerPlus model on KITTI:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files
```
python PixelFormerPlus/train.py arguments_train_kittieigen.txt
```
Training the PixelFormer model on HabitatDyn Validation set 1:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files  
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
```
python PixelFormerPlus/train.py arguments_train_habitat.txt
```

Training the PixelFormer model on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
```
python PixelFormerPlus/train.py arguments_train_habitat.txt
```
Training the PixelFormerPlus model on HabitatDyn Validation set 1:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
```
python PixelFormerPlus/train.py arguments_train_habitat.txt
```

Training the PixelFormerPlus model on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
```
python PixelFormerPlus/train.py arguments_train_habitat.txt
```
Training the SemanticBins model on HabitatDyn Validation set 1:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
```
python PixelFormerPlus/train_semantic.py arguments_train_habitat.txt
```

Training the SemanticBins model on HabitatDyn Validation set 2:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
```
python PixelFormerPlus/train_semantic.py arguments_train_habitat.txt

## Evaluation
Evaluate the PixelFormer model on NYUv2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/PixelFormer_NYU
```
python PixelFormerPlus/eval.py arguments_eval_nyu.txt
```

Evaluate the PixelFormer model on KITTI:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/PixelFormer_KITTI
```
python PixelFormerPlus/eval.py arguments_eval_kitti.txt
```
Evaluate the PixelFormerPlus model on NYUv2:  
Rename the network folder of PixelFormerPlus model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/NewModel_NYU
```
python PixelFormerPlus/eval.py arguments_eval_nyu.txt
```

Evaluate the PixelFormerPlus model on KITTI:  
Rename the network folder of PixelFormerPlus model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/NewModel_KITTI
```
python PixelFormerPlus/eval.py arguments_eval_kitti.txt
```
Evaluate the PixelFormer model on HabitatDyn Validation set 1:
Rename the network folder of PixelFormer model from networkssource to networks    
Modify the datapaths of ground truth and raw data in the config files  
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/PixelFormer_Validation1
```
python PixelFormerPlus/eval.py arguments_eval_habitat.txt
```

Evaluate the PixelFormer model on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/PixelFormer_Validation2
```
python PixelFormerPlus/eval.py arguments_eval_habitat.txt
```
Evaluate the PixelFormerPlus model on HabitatDyn Validation set 1:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/NewModel_Validation1
```
python PixelFormerPlus/eval.py arguments_eval_habitat.txt
```

Evaluate the PixelFormerPlus model on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/NewModel_Validation2
```
python PixelFormerPlus/eval.py arguments_eval_habitat.txt
```
Evaluate the SemanticBins model on HabitatDyn Validation set 1:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/SemanticBins_Validation1
```
python PixelFormerPlus/eval_semantic.py arguments_eval_habitat.txt
```

Evaluate the SemanticBins model on HabitatDyn Validation set 2:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/SemanticBins_Validation2
```
python PixelFormerPlus/eval_semantic.py arguments_eval_habitat.txt
## Testing
Test the PixelFormer model and get depth maps on NYUv2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/PixelFormer_NYU
```
python PixelFormerPlus/test.py arguments_test_nyu.txt
```

Test the PixelFormer model and get depth maps on KITTI:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/PixelFormer_KITTI
```
python PixelFormerPlus/text.py arguments_test_kitti.txt
```
Test the PixelFormerPlus model and get depth maps on NYUv2:  
Rename the network folder of PixelFormerPlus model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/NewModel_NYU
```
python PixelFormerPlus/test.py arguments_eval_test.txt
```

Test the PixelFormerPlus model and get depth maps on KITTI:  
Rename the network folder of PixelFormerPlus model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files
Set the parameter checkpoint_path to pretrained/NewModel_KITTI
```
python PixelFormerPlus/test.py arguments_test_kitti.txt
```
Test the PixelFormer model and get depth maps on HabitatDyn Validation set 1:
Rename the network folder of PixelFormer model from networkssource to networks    
Modify the datapaths of ground truth and raw data in the config files  
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/PixelFormer_Validation1
```
python PixelFormerPlus/test.py arguments_test_habitat.txt
```

Test the PixelFormer model and get depth maps on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormer model from networkssource to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/PixelFormer_Validation2
```
python PixelFormerPlus/test.py arguments_test_habitat.txt
```
Test the PixelFormerPlus model on HabitatDyn Validation set 1:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/NewModel_Validation1
```
python PixelFormerPlus/test.py arguments_test_habitat.txt
```

Test the PixelFormerPlus model and get depth maps on HabitatDyn Validation set 2:  
Rename the network folder of PixelFormerPlus model to networks  
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/NewModel_Validation2
```
python PixelFormerPlus/test.py arguments_test_habitat.txt
```
Test the SemanticBins model and get depth maps on HabitatDyn Validation set 1:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val0102scenes.txt
Set the parameter checkpoint_path to pretrained/SemanticBins_Validation1
```
python PixelFormerPlus/test_semantic.py arguments_test_habitat.txt
```

Test the SemanticBins model and get depth maps on HabitatDyn Validation set 2:    
Modify the datapaths of ground truth and raw data in the config files 
Set the parameter filenames_file to data_splits/val1.txt  
Set the parameter filenames_file_eval data_splits/val16181619clips.txt
Set the parameter checkpoint_path to pretrained/SemanticBins_Validation2
```
python PixelFormerPlus/test_semantic.py arguments_test_habitat.txt
## Pretrained Models
* You can download the pretrained models from [here](https://drive.google.com/drive/folders/1CupMODnxBe4JJ15jbet3nz2xgTjL5ExM?usp=drive_link)




