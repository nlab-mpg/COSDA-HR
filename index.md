# What is COSDA?

The object recognition ability is indispensable for robotsto act like humans in a home environment. For example, when considering an object searching task, humans can recognize a naturally arranged object previously held intheir hands while ignoring never observed objects. Even in such a simple task, we need to deal with three complex problems: **Domain Adaptation**, **Open-Set Recognition**, and **Continual Learning**. However, most existing datasets are simplified to focus on one problem and do not measure the object recognition ability for home robots when multiple problems are mixed in. We propose a Continual Open Set Domain Adaptation (COSDA) Dataset that requires dealing with the above three problems simultaneously.


# Dataset

COSDA, specifically designed for (C)ontinual (O)pen-(S)et (D)omain (A)daptation. All images in the COSDA dataset are captured by theXtion RGB-D sensor mounted at the eye-level of the Toyota Human Support Robot (HSR) whose appearanceand specifications are shown in the supplementary material. In addition to RGB images, depth information is alsoavailable in our dataset although we do not use it in this paper. There are 16 super categories and each category has different 10 instances, thus there are  160 classes in total. The 16 super categories are selected from those commonly appear in a home environment which include ball, book, bowl, toy block, can, cup, dish, glass bottle, mobile phone,pen, plastic bottle, plush doll, TV controller, scissors, socks, and towel.
<div style="text-align:center">
<img src="cosda/image/example.png" alt="" width="500" height="500"/>
</div>


# How to download?

Click [Download .tar.gz] button to download the dataset.


# How to use?
After decompressing the file, you would see these folders.
```
data/
    |----train
    |----train_target_infer
    |----val
    |----val_target_infer
```

**train**: 
```
|----000
|----001
|----002
|----003
   ...
|----159
```

**train_target_infer**:

```
/
|- midday/
|    |- case1/
|    |    |- pattern1_night_0396.bmp
|    |    |- pattern1_night_0396.bmp
|    |    |- ...
|    |
|    |- case2/
|    |    |- ...
|    | 
|    |- ...
|    
|- night/
|    |- case1/

|    |- ...
|
|- ...
|
|- 161/
     |- pattern1_night_0396_0000.png
     |- ...
```

"""
    This data was taken from no_annotation_avi data on GoogleDrive
"""

All inference was using the parameters which was tuned on
the pattern1 and pattern2_night.

################################

result on tuning (pattern1 and pattern2_night)
    mean_iou  : 0.7273772954940796
    precision : 0.015073592437489286
    recall    : 0.96696452748443

IoU was calculated only on matched GT bbox.
Precision was calculated by (GT bbox)/(pred bbox including matched bbox).
Recall was calculate by (matched bbox)/(GT bbox).

bbox matching was done by thresholding iou >= 0.5 and best IoU.

################################
image name : <original_image_name>_<detection_num>.png styled.


**val**:
```
|----000
|----001
|----002
|----003
   ...
|----159
```


**val_target_infer**:
```
|----000
|----001
|----002
|----003
   ...
|----159
|----160
```


# Benchmark

<div style="text-align:center">
<img src="cosda/image/benchmark.png" alt="" width="1100" height="160"/>
</div>
