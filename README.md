Enhancing Ovarian Follicle Assessment through Distribution Free Risk Control and Contextual Information in Deep Learning Models
====================================================

🔗 Requirements
===============
Python 3.8+ 


Install the required packages:

- Via poetry:
```
$ poetry init
```

📚 Data Availability
===============

Download the file [here](https://drive.google.com/file/d/1Ry0w6sU22-kpv2Z31HfSYUEFlvKYuXkO/view?usp=sharing) and then run 

```
$ unzip ovary_cut.zip
```

Once the file is unzipped, data is organized a follows:


    ├── ...
    └── follicle-risk-control
       ├── 01_ovary_cuts                    
       │   ├── ovary_images
       |   |   ├──   0A_a/0A_a__roi0.tif
       |   |   ├──   0A_a/0A_a__roi1.tif
       |   |   ├──   ...
       |   |   ├──   2A_g/2A_g__roi0.tif
       |   |   ├──   2A_g/2A_g__roi1.tif
       |   |   ├──   2A_g/2A_g__roi2.tif
       |   |   ├──   ...
       │   └── ovaries_annotations.json    
       └── 03_model_weights
               └── efficientdet
                   └──   */*.ckpt

✂️ Create patches and annotations to train OD models
===============
Patches are only going to be generated for mouse cuts belonging to training and validation set as for the calibration and test sets the slicing of the ovary cuts into patches will be done on the fly. The generated patches are of size $1000 \times 1000$ with a stride of $512$. Only 5% of patches without any follicle are saved.

```
$ python main/generate_patches_stride_train_val.py
```

Patches are saved in `02_model_inputs/patches` and corresponding annotations in `02_model_inputs/annotations_efficientdet/patches_annotation_stride_train_val.json` in Pascal VOC data format.

🏋️‍♀️ Train models
===============

To train OD algorithms run:

```
$ python main/train_efficientdet.py
$ python main/train_yolo.py
```

To create the dataset for classifcation run:

```
$ python main/create_classif_train_dataset.py
```

To train the classification model:

```
$ python main/train_classif.py
```

If you don't want to retrain the models, weights are available in the zip file, in the 03_model_weights folder.


📈 Inference
===============
Inference is run at the ovary cut level. To run the EfficientDet and Yolo inferences run

```
$ python main/predict_effdet_with_depth_and_classif.py
$ python main/predict_yolo_with_depth_and_classif.py
```

The output of the inference is a `json` file of the following shape:

```
{
    "OA_c": {
        "roi0": {
            "bboxes": [[xmin, ymin, xmax, ymax], ...],
            "scores": [.5, ...],
            "depths": [.3, ...],
            "scores_classif": [.6, ...],
            "classes": ["PMF", ...]
        },
        "roi1": {...},
        ...
    },
    ...,
    "4A_c": {
        "roi0": {...},
        ...
    },
    ...
}
```

Inference files for both model are saved in their respective directories with the depth of each prediction already computed: `04_model_output/efficientdet/result.json`

✅ Run the LTT procedure
===============
The LTT procedure can be run with the notebooks : `notebooks/analyse_ltt_effdet.ipynb` and `notebooks/analyse_ltt_yolo.ipynb`
