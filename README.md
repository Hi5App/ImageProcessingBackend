
3D U-Net (https://arxiv.org/abs/1606.06650) is adopted to buid a segmentation network and generate segmentations for several critical brain regions.

# Dependencies
To run the inference scripts, several dependencies are required to be installed.

The user can install the dependencies directly by running:

```shell
  pip install -r requirements.txt
```

# Inference

The images for segmentation need to be placed at `./data/input`. The weight is stored at `./logs`, which is obtained by training on multiple fmost modal data. 
The following script is used to segment the images:

```shell
  python predict_unet.py
```

Inference the 3D U-Net with the default parameters would require a GPU with more than 12GB of RAM.
Paramters such as the model depth and the input size can be modified in `predict_unet.py`.

The results will be stored at `./data/predict/[filename]`. For each input image, several result files are generated, including `seg.v3draw`, `0.v3draw`, `1.v3draw`, ..., `8.v3draw`.

Optionally, the resulting `v3draw` files can be visualized using [Vaa3D](https://vaa3d.org).


