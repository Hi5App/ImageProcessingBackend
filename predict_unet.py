import os
import sys
import numpy as np
import SimpleITK as sitk
from numba import njit
from scipy.ndimage.interpolation import zoom
import tensorflow as tf
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
config["val_file"] = os.path.abspath("./DataPreprocess/val.h5")
config["model_file"] = os.path.abspath("./logs/U_model.h5")
config["overwrite"] = False  # If False, do not load model.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def itensity_normalize_one_volume(volume):
#     pixels = volume[volume > 0]
#     mean = pixels.mean()
#     std = pixels.std()
#     out = (volume - mean) / std
#     out_random = np.random.normal(0, 1, size=volume.shape)
#     out[volume == 0] = out_random[volume == 0]
#     return out

@njit
def itensity_normalize_one_volume(volume):
    # Find indices where volume is greater than zero
    indices = np.nonzero(volume > 0)

    # Extract non-zero pixels using a loop
    non_zero_pixels = []
    for i in range(len(indices[0])):
        non_zero_pixels.append(volume[indices[0][i], indices[1][i], indices[2][i]])
    non_zero_pixels = np.array(non_zero_pixels, dtype=np.float32)

    if non_zero_pixels.size == 0:
        mean = 0.0
        std = 1.0
    else:
        mean = non_zero_pixels.mean()
        std = non_zero_pixels.std()

    # Normalize the volume
    out = (volume.astype(np.float32) - mean) / std

    # Generate random noise
    out_random = np.random.normal(0, 1, size=volume.shape).astype(np.float32)

    # Replace zero values with random noise
    zero_indices = np.nonzero(volume == 0)
    for i in range(len(zero_indices[0])):
        out[zero_indices[0][i], zero_indices[1][i], zero_indices[2][i]] = out_random[
            zero_indices[0][i], zero_indices[1][i], zero_indices[2][i]]

    return out


# def ResizeData(data, InputShape):
#     [W, H, D] = data.shape
#     scale = [InputShape[0] * 1.0 / W, InputShape[1] * 1.0 / H, InputShape[2] * 1.0 / D]
#     data = zoom(data, scale, order=1)
#     return data

def ResizeData(data, InputShape):
    [W, H, D] = data.shape
    original_depth = data.shape[2]

    # 初始化一个用于存储调整大小后的数据的数组
    resized_data = np.zeros(InputShape, dtype=data.dtype)

    # 对每个深度切片进行处理
    for z in range(InputShape[2]):
        # 计算在原始深度中对应的索引
        original_z = int(z / InputShape[2] * original_depth)
        # 转换为Tensor，并添加一个额外的维度来形成3维张量
        data_slice = tf.convert_to_tensor(data[:, :, original_z])[..., tf.newaxis]
        # 使用TensorFlow的resize函数
        resized_slice = tf.image.resize(data_slice, InputShape[0:2], method=tf.image.ResizeMethod.BILINEAR)
        # 移除添加的维度，并将结果保存到输出数组中
        resized_data[:, :, z] = resized_slice[..., 0].numpy()

    return resized_data


# def ResizeMap(data, InputShape):
#     [W, H, D, C] = data.shape
#     scale = [InputShape[0] * 1.0 / W, InputShape[1] * 1.0 / H, InputShape[2] * 1.0 / D, 1]
#     data = zoom(data, scale, order=1)
#     return data

def ResizeMap(data, InputShape):
    # 获取原始数据的深度和通道数
    original_depth = data.shape[2]
    num_channels = data.shape[3]

    # 初始化一个用于存储调整大小后的数据的数组
    resized_data = np.zeros(InputShape)

    # 对每个通道和每个切片进行处理
    for c in range(num_channels):
        for z in range(InputShape[2]):
            # 计算在原始深度中对应的索引
            original_z = int(z / InputShape[2] * original_depth)
            # 转换为Tensor，并添加一个额外的维度来形成3维张量
            data_slice = tf.convert_to_tensor(data[:, :, original_z, c])[..., tf.newaxis]
            # 使用TensorFlow的resize函数
            resized_slice = tf.image.resize(data_slice, InputShape[0:2], method=tf.image.ResizeMethod.BILINEAR)
            # 移除添加的维度，并将结果保存到输出数组中
            resized_data[:, :, z, c] = resized_slice[..., 0].numpy()

    return resized_data


def save_image(im_save, pre_hot, i, save_dir, name, v3d_flag):
    if v3d_flag:
        im_save['data'] = pre_hot[:, :, :, i][..., np.newaxis]
        im_save['data'] = np.uint8(im_save['data'] * 255)
        im_save['data'].flags['WRITEABLE'] = True
        save_path = os.path.join(save_dir, name.split('.v3draw')[0], f"{i}.v3draw")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if i == 9:
            save_path = os.path.join(save_dir, name.split('.v3draw')[0], "8.v3draw")
        save_v3d_raw_img_file1(im_save, save_path)
    else:
        im_save = sitk.GetImageFromArray(pre_hot[:, :, :, i])
        save_path = os.path.join(save_dir, name.split('.nii')[0], f"{i}.nii")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if i == 9:
            save_path = os.path.join(save_dir, name.split('.nii')[0], "8.nii")
        sitk.WriteImage(im_save, save_path)


def save_segmentation(im_save, pre_class, save_dir, name, v3d_flag):
    if v3d_flag:
        im_save['data'] = pre_class[..., np.newaxis]
        im_save['data'] = np.uint8(im_save['data'])
        save_path = os.path.join(save_dir, name.split('.v3draw')[0], 'seg.v3draw')
        save_v3d_raw_img_file1(im_save, save_path)
    else:
        im_save = sitk.GetImageFromArray(pre_class)
        save_path = os.path.join(save_dir, name.split('.nii')[0], 'seg.nii')
        sitk.WriteImage(im_save, save_path)


def inference(image_name):
    save_dir = '../inferenceimagepath/predict/'
    image_dir = '../inferenceimagepath/input/'
    model_dir = config["model_file"]

    print("is_built_with_cuda: ", tf.test.is_built_with_cuda())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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
                print(f'processing class {i} result...')
                pre_class[pre_class == i] = class2inten[i]
                if i != 8:
                    save_image(im_save, pre_hot, i, save_dir, name, v3d_flag)

            print('saving seg result...')
            save_segmentation(im_save, pre_class, save_dir, name, v3d_flag)
            print('image processed successfully')
            return {'status': 'ok', 'message': 'image processed successfully'}

            # for i in range(10):
            #     print('processing class ', i, ' result...')
            #     # processing classes prediction
            #     pre_class[pre_class == i] = class2inten[i]
            #     # saving one hot
            #     if i != 8:
            #         if v3d_flag:
            #             im_save['data'] = pre_hot[:, :, :, i]
            #             im_save['data'] = im_save['data'][..., np.newaxis]
            #
            #             # Convert the data to 8-bit integer
            #             im_save['data'] = np.uint8(im_save['data'] * 255)
            #
            #             im_save['data'].flags['WRITEABLE'] = True
            #             save_path = save_dir + name.split('.v3draw')[0] + '/'
            #             if not os.path.exists(save_path):
            #                 os.mkdir(save_path)
            #             if i != 9:
            #                 save_path = save_path + str(i) + '.v3draw'
            #             elif i == 9:
            #                 save_path = save_path + '/8.v3draw'
            #             save_v3d_raw_img_file1(im_save, save_path)
            #         else:
            #             # TODO：nii format dimension?
            #             im_save = sitk.GetImageFromArray(pre_hot[:, :, :, i])
            #             save_path = save_dir + name.split('.nii')[0] + '/'
            #             if not os.path.exists(save_path):
            #                 os.mkdir(save_path)
            #             if i != 9:
            #                 save_path = save_path + str(i) + '.nii'
            #             elif i == 9:
            #                 save_path = save_path + '/8.nii'
            #             sitk.WriteImage(im_save, save_path)
            #
            # # saving classes prediction
            # print('saving seg result...')
            # if v3d_flag:
            #     im_save['data'] = pre_class[..., np.newaxis]
            #
            #     # Convert the data to 8-bit integer
            #     im_save['data'] = np.uint8(im_save['data'])
            #
            #     save_path = save_dir + name.split('.v3draw')[0] + '/seg.v3draw'
            #     save_v3d_raw_img_file1(im_save, save_path)
            # else:
            #     im_save = sitk.GetImageFromArray(pre_class)
            #     save_path = save_dir + name.split('.nii')[0] + '/seg.nii'
            #     sitk.WriteImage(im_save, save_path)
            # print('image processed successfully')
            # return {'status': 'ok', 'message': 'image processed successfully'}
        if isImageFound:
            break

    if isImageFound is False:
        print('image not found, please check image name')
        return {'status': 'error', 'message': 'image not found, please check image name'}


if __name__ == "__main__":
    inference("191815.v3draw")
