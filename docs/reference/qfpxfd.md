# Readme

`{code_root_dir}` represents the `code` folder in our uploaded model.

> Attention: This document uses Keras backend on tensorflow. You can refer to https://keras-cn.readthedocs.io/en/latest/layers/core_layer/ for more details.

## A. Training

### 1. Data Preparation

```
cd {code_root_dir}/data_builder
bash build_train_dataset.sh # change paths to run properly
```

**data_builder** is the module to build the dataset required for training and test. The public dataset we have used for training are [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI;jsessionid=266E6ABF5C1BEF5CAC8B09EA285C9724), [DSB](https://www.kaggle.com/c/data-science-bowl-2017/data), [SPIE](https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge). All the data from those dataset would be converted to a hdf5 file named 'vol.hdf5' and we upload files named as 'label_dict.npy' which contain our labels.

The easy-to-use script is 'build_dataset.sh', but you have to change the corresponding directories where dicom files store.

The following description illustrates how to manually run commands in the script, which is necessary if you want to modify the script. `build_dataset.py` and `flip_dataset.py` are main programs. `dicom_parser.py` is the helper.

```
python build_dataset.py -i $input_dir$ -o $output_dir$ -m $mode$ -v $version$
```

- LIDC-IDRI: 

  Specially, for LIDC-IDRI, we have 2 versions, v9 and v15. The main difference between two versions is that v9 converts dicom files ordered by 'SliceLocation' or 'ImagePosition', while v15 ordered by 'InstanceNumber'. Accordingly, the labels also differ slightly.

  ```
  # '*/LIDC-IDRI/DOI' is the raw dicom files
  python build_dataset.py -i '*/LIDC-IDRI/DOI' -o '../../data/lidc_v9' -m 'lidc' -v 'v9'
  python build_dataset.py -i '*/LIDC-IDRI/DOI' -o '../../data/lidc_v15' -m 'lidc' -v 'v15'
  ```

- DSB:

  DSB stage1 data is used. We manually check the data to ensure that all the CT scans are from head to foot. Thus, a list to flip data is offered as 'flip_list.npy'. After converting dicom files of Kaggle stage1, a script to flip 'vol.hdf5' is required to be executed.

  ```
  # '*/stage1' is the raw dicom files
  # '-v' option is omitted
  python build_dataset.py -i '*/stage1' -o '../../data/kaggle' -m 'kaggle'
  python flip_dataset.py -d '../../data/kaggle/vol.hdf5'
  ```

- SPIE-AAPM:

  ```
  # '*/SPIE/DCM' is the raw dicom files
  # '-v' option is omitted
  python build_dataset.py -i '*/stage1' -o '../../data/spie' -m 'spie'
  ```

#### 1.1 Flip Regression

```
cd {code_root_dir}/flip_regression
flip_train.sh
```

**flip_regression** is the module to flip the CT scans to make sure that the scans are from head to foot. If `code/data_builder/build_train_dataset.sh` is executed, this part could be ignored.

Please use those scripts after the kaggle stage1 data has been converted and flipped, because it has to be used for training.

#### 1.2 Lung Segmentation

   ```
   cd {code_root_dir}/lung_segmentation
   python -W ignore LungSegmentationParallel.py 
   ```

### 2. Nodule Detection

- Setup Required Environment

  - install `Anaconda2 (python2.7)`
  - change the directory to `{code_root_dir}/faster_rcnn_candidates/v9/caffe-fast-rcnn`
  - make caffe and setup caffe path (follow the setup instructions on (we don't use cuDNN): https://github.com/rbgirshick/py-faster-rcnn)

  ```
  cd {code_root_dir}/faster_rcnn_candidates/v9
  cd lib
  make
  cd ..
  cd caffe-fast-rcnn
  make -j8 && make pycaffe
  ```

- edit `~/.bashrc` and add the caffe path to `PYTHONPATH`

  ```
  export PYTHONPATH={code_root_dir}/caffe-fast-rcnn/python:$PYTHONPATH
  ```


- Prepare Faster R-CNN training data

  ```
  cd {code_root_dir}/cr_frcnn_data
  python cr_data_v9.py
  python cr_data_v15.py
  ```


- For Training

  ```
  cd {code_root_dir}/faster_rcnn_candidates/v9
  python train_test.py
  cd {code_root_dir}/faster_rcnn_candidates/v15
  python train_test.py
  ```

#### 2.1 False Positive Reduction

```
cd {code_root_dir}/fp_reduction
bash kaggle_train.sh
```

### 3. Nodule Malignancy Classification

```
cd {code_root_dir}/cassification_branch1
bash train.sh
```

```
cd {code_root_dir}/cassification_branch2
bash vgg13_train.sh
bash vgg13_cv.sh
```

### 4. Model Ensemble

```
{code_root_dir}/submission
pyhton merge_main.py
```

> Note: The easy-to-use script we provide is only to run the pipeline sequentially. However, you might run the same code with different dataset and GPUs in parallel to speed up the training process. Then, you could check our bash scripts, and run the commands manually, with some minor changes like the index of GPU to use.

## B. Prediction

### 1. Data Preparation & Flip Regression

```
cd {code_root_dir}/data_builder
bash build_test_dataset.sh # include flip regression
```

The easy-to-use script is `build_test_dataset.sh`, but you have to change the corresponding directories where dicom files store.

The program is assumed to receive a path, each children fold of which corresponds to a series of CT scans, just like stage1.

```
# '*/stage2' is the raw dicom files
# '-v' option is omitted
python build_dataset.py -i '*/stage2' -o '../../data/stage2' -m 'kaggle'
```

### 2. Lung Segmentation 

```
cd {code_root_dir}/lung_segmentation
# For stage2 testing
python -W ignore LungSegmentationParallelForTest.py
```

### 3. Candidate Detection

- Setup Required Environment

  - install `Anaconda2 (python2.7)`
  - change the directory to `{code_root_dir}/faster_rcnn_candidates/v9/caffe-fast-rcnn`
  - make caffe and setup caffe path (follow the setup instructions on (we don't use cuDNN): https://github.com/rbgirshick/py-faster-rcnn)

  ```
  cd {code_root_dir}/faster_rcnn_candidates/v9
  cd lib
  make
  cd ..
  cd caffe-fast-rcnn
  make -j8 && make pycaffe
  ```

  - edit `~/.bashrc` and add the caffe path to `PYTHONPATH`

  ```
  export PYTHONPATH={code_root_dir}/caffe-fast-rcnn/python:$PYTHONPATH
  ```

  - run the python script `predict.py` with additional arguments
    - --gpu: set the gpu id
    - --start_idx: the starting scan index
    - --end_idx: the ending scan index, if it is equal to -1, then, it is set as the last scan index

  ```
  cd {code_root_dir}/faster_rcnn_candidates/v9
  python predict.py --gpu 0 --start_idx 0 --end_idx -1
  ```

  - run the python script `predict.py` with additional arguments, these arguments is the same meaning as above

  ```
  cd {code_root_dir}/faster_rcnn_candidates/v15
  python predict.py --gpu 0 --start_idx 0 --end_idx -1
  ```

### 4. False Positive Reduction

```
cd {code_root_dir}/fp_reduction
bash kaggle_predict.sh
```

### 5. Nodule Malignancy Classification

```
cd {code_root_dir}/cassification_branch1
bash bash predict_stage2.sh
```

```
cd {code_root_dir}/cassification_branch2
bash vgg13_predict.sh
```

### 6. Model Ensemble

```
python scanScoreSubmission.py
```
