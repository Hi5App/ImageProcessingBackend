import os
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

from model import unet_model_3d
from Formatcov import load_v3d_raw_img_file1, save_v3d_raw_img_file1
import argparse

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument("--nii", action='store_true')
parser.add_argument("--gpu", '-g', default='0', type=str)
args = parser.parse_args()

config = dict()
config["pool_size"] = (2, 2, 2)  # pool size for the max pooling operations
config["image_shape"] = (80, 144, 112)  # This determines what shape the images will be cropped/resampled to.
# orishape: input_shape = (200, 324, 268)
config["n_labels"] = 10
config["input_shape"] = tuple(list(config["image_shape"]) + [1])
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution
config["batch_size"] = 1
config["validation_batch_size"] = config["batch_size"]
config["n_epochs"] = 200  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 15  # training will be stopped after this many epochs without the validation loss improving
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["train_file"] = os.path.abspath("./DataPreprocess/train.h5")
config["val_file"] = os.path.abspath("./DataPreprocess//val.h5")
config["model_file"] = os.path.abspath("./logs/U_model.h5")
config["overwrite"] = False  # If False, do not load model.

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def itensity_normalize_one_volume(volume):
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    out_random = np.random.normal(0, 1, size=volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out


def ResizeData(data, InputShape):
    [W, H, D] = data.shape
    scale = [InputShape[0] * 1.0 / W, InputShape[1] * 1.0 / H, InputShape[2] * 1.0 / D]
    data = zoom(data, scale, order=1)
    return data


def ResizeMap(data, InputShape):
    [W, H, D, C] = data.shape
    scale = [InputShape[0] * 1.0 / W, InputShape[1] * 1.0 / H, InputShape[2] * 1.0 / D, 1]
    data = zoom(data, scale, order=1)
    return data


def inference(image_name):
    save_dir = 'data/predict/'
    image_dir = 'data/input/'
    model_dir = config["model_file"]

    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    model = unet_model_3d(input_shape=config["input_shape"],
                          pool_size=config["pool_size"],
                          n_labels=config["n_labels"],
                          deconvolution=config["deconvolution"])
    if os.path.exists(model_dir):
        model.load_weights(model_dir, by_name=True)
        print(' ------------  load model !')
    else:
        print('model do not existing in ', model_dir)
        return
    input_shape = config["image_shape"]

    class2inten = {}
    class2inten[0] = 0
    class2inten[1] = 62
    class2inten[2] = 75
    class2inten[3] = 80
    class2inten[4] = 100
    class2inten[5] = 145
    class2inten[6] = 159
    class2inten[7] = 168
    class2inten[8] = 0
    class2inten[9] = 249

    names = os.listdir(image_dir)
    isImageFound = False
    for name in names:
        if image_name == name:
            isImageFound = True
            image_path = os.path.join(image_dir, name)
            if name.endswith('.v3draw'):
                v3d_flag = True
                im_v3d = load_v3d_raw_img_file1(image_path)['data']
                shape_v3d = im_v3d.shape
                im_np = im_v3d[..., 0]
                ori_shape = im_np.shape
                im_np = ResizeData(im_np, input_shape)
                im_np = itensity_normalize_one_volume(im_np)
                im_np = im_np[np.newaxis, ..., np.newaxis]
                im_save = {}
                im_save['size'] = shape_v3d
                im_save['datatype'] = 1
            elif name.endswith('.nii'):
                # nii should have size: 320 568 456
                # TODO: only handle 456 320 568 like issue
                v3d_flag = False
                im_np = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
                ori_shape = im_np.shape
                if ori_shape[-1] > ori_shape[0] and ori_shape[-1] > ori_shape[1]:
                    print('transfer nii shape ', ori_shape)
                    im_np = im_np.transpose(1, -1, 0)
                    ori_shape = im_np.shape
                    print('to ', ori_shape)
                im_np = ResizeData(im_np, input_shape)
                im_np = itensity_normalize_one_volume(im_np)
                im_np = im_np[np.newaxis, ..., np.newaxis]
            else:
                print('wrong image format')
                return

            print('input image has shape: ', ori_shape, ' and will resize to shape: ', input_shape, ' as model input')

            print('--------------------- predicting: ', name, '... ----------------------------')
            pre_hot = model.predict(im_np)[0]

            # saving segmentation map to nii or v3draw
            print('resize predict map...')
            pre_hot = ResizeMap(pre_hot, ori_shape + (10,))
            print(pre_hot.shape)
            pre_class = np.float32(np.argmax(pre_hot, axis=3))

            for i in range(10):
                print('processing class ', i, ' result...')
                # processing classes prediction
                pre_class[pre_class == i] = class2inten[i]
                # saving one hot
                if i != 8:
                    if v3d_flag:
                        im_save['data'] = pre_hot[:, :, :, i]
                        im_save['data'] = im_save['data'][..., np.newaxis]

                        # Convert the data to 8-bit integer
                        im_save['data'] = np.uint8(im_save['data'] * 255)

                        im_save['data'].flags['WRITEABLE'] = True
                        save_path = save_dir + name.split('.v3draw')[0] + '/'
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        if i != 9:
                            save_path = save_path + str(i) + '.v3draw'
                        elif i == 9:
                            save_path = save_path + '/8.v3draw'
                        save_v3d_raw_img_file1(im_save, save_path)
                    else:
                        # TODO：nii format dimension?
                        im_save = sitk.GetImageFromArray(pre_hot[:, :, :, i])
                        save_path = save_dir + name.split('.nii')[0] + '/'
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        if i != 9:
                            save_path = save_path + str(i) + '.nii'
                        elif i == 9:
                            save_path = save_path + '/8.nii'
                        sitk.WriteImage(im_save, save_path)

            # saving classes prediction
            print('saving seg result...')
            if v3d_flag:
                im_save['data'] = pre_class[..., np.newaxis]

                # Convert the data to 8-bit integer
                im_save['data'] = np.uint8(im_save['data'])

                save_path = save_dir + name.split('.v3draw')[0] + '/seg.v3draw'
                save_v3d_raw_img_file1(im_save, save_path)
            else:
                im_save = sitk.GetImageFromArray(pre_class)
                save_path = save_dir + name.split('.nii')[0] + '/seg.nii'
                sitk.WriteImage(im_save, save_path)
            print('image processed successfully')
            return {'status': 'ok', 'message': 'image processed successfully'}

    if isImageFound is False:
        print('image not found, please check image name')
        return {'status': 'error', 'message': 'image not found, please check image name'}


if __name__ == "__main__":
    inference("test.v3draw")
