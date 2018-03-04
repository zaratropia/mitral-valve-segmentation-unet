from __future__ import print_function

import os
import re
import numpy as np
import nrrd
import json
import simplejson
import png

from skimage.io import imsave, imread
from skimage.transform import resize
from PIL import Image

data_path = 'raw/'


def min_image_rows():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    min_rows = 1000
    for image_name in images:
        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        if min_rows > img.shape[0]:
            min_rows = img.shape[0]
    return min_rows

def min_image_cols():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    min_cols = 1000
    for image_name in images:
        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        if min_cols > img.shape[0]:
           min_cols = img.shape[1]
    return min_cols

image_rows = min_image_rows()
image_cols = min_image_cols()
# old approach
# image_rows = # max_image_rows()
# image_cols = max_image_cols()

def create_resized_train_data():
    image_rows = max_image_rows()
    image_cols = max_image_cols()

    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, image_rows, image_cols))
    imgs_mask = np.ndarray((total, image_rows, image_cols))

    i = 0
    print('*'*30)
    print('Creating training images...')
    print('*'*30)
    for image_name in images:
        image_mask_name = image_name.split('.')[0] + '_mask.nrrd'
        # skip masks
        if 'mask' in image_name or os.path.isdir(image_name):
            continue
        # skip broken masks
        if not os.path.isfile(os.path.join(train_data_path, image_mask_name)):
            continue

        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        img_mask, head = nrrd.read(os.path.join(train_data_path, image_mask_name))

        img = img.squeeze()
        img_mask = img_mask.squeeze()

        imgs[i] = resize(img, (image_rows, image_cols), preserve_range=True)
        # Process binary masks
        img_mask[img_mask > 1] = 1
        imgs_mask[i] = resize(img_mask, (image_rows, image_cols), preserve_range=True)

        # if i % 100 == 0:
        #     print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def create_cropped_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2
    print(total)
    imgs = np.ndarray((total, image_rows, image_cols))
    imgs_mask = np.ndarray((total, image_rows, image_cols))

    i = 0
    print('*'*30)
    print('Creating training images...')
    print('*'*30)
    for image_name in images:
        image_mask_name = image_name.split('.')[0] + '_mask.nrrd'
        # skip masks
        if 'mask' in image_name or os.path.isdir(image_name):
            continue
        # skip broken masks
        if not os.path.isfile(os.path.join(train_data_path, image_mask_name)):
            continue

        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        img_mask, head = nrrd.read(os.path.join(train_data_path, image_mask_name))

        img = img.squeeze()
        img_mask = img_mask.squeeze()

        imgs[i] = crop_center(img, image_cols, image_rows)
        # Process binary masks
        img_mask[img_mask > 1] = 1
        imgs_mask[i] = crop_center(img_mask, image_cols, image_rows)

        # if i % 100 == 0:
        #     print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')

# Crop helper
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def preprocess_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)

    i = 0
    print('*'*30)
    print('Process test images...')
    print('*'*30)
    for image_name in images:
        filename = os.path.join(train_data_path, image_name)
        if "_mask" in filename:
            os.remove(filename)


def preprocess_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    i = 0
    print('*'*30)
    print('Process train images...')
    print('*'*30)
    for image_name in images:
       filename = os.path.join(train_data_path, image_name)
       if "_mask" in filename:
          ndarray_slice, head = nrrd.read(filename)
          if ndarray_slice.max() == 0:
             print ('Empty mask - delete: ' + filename)
             # remove only if blank mask, than remove also normal image
             os.remove(filename)
             os.remove(filename.replace('_mask', ''))


def cleanup_broken_data():
    cleanup_filter = [range(5, 35)]
    for x in cleanup_filter:
        if os.path.exists(data_path + 'broken/' + str(x)):
            broken_data_path = os.path.join(data_path, 'broken/', str(x))
            images = os.listdir(broken_data_path)
            for image_name in images:
                # remove mask anhd image from train pool
                filename = os.path.join(os.path.join(data_path, 'train'), image_name)
                drop_file = filename.replace('tiff', 'nrrd')
                os.remove(drop_file)
                os.remove(drop_file.replace('_mask', ''))


def create_test_data():
    test_data_path = os.path.join(data_path, 'test')
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols))
    imgs_id = np.ndarray((total, ), dtype=object) # object to store filename
    imgs_header = []

    i = 0
    print('*'*30)
    print('Creating test images...')
    print('*'*30)
    for image_name in images:
        if "mask" in image_name:
            continue

        size = len(image_name.split('.'))
        img_id = image_name.split('.')[size-2].replace("_mask", "")
        img, head = nrrd.read(os.path.join(test_data_path, image_name))
        head["custom_name"] = image_name;

        img = img.squeeze()

        imgs[i] = crop_center(img, image_cols, image_rows)
        # or imgs[i] = resize(img, (image_rows, image_cols), preserve_range=True)
        imgs_id[i] = img_id
        imgs_header.append(head)

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')
    with open('imgs_header_test.json', 'wb') as outfile:
        simplejson.dump(imgs_header, outfile)
    print('Saving header information done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    with open('imgs_header_test.json', 'r') as jsonfile:
        imgs_header = json.loads(jsonfile.read())
    return imgs_test, imgs_id, imgs_header


# def test():
#     print(image_rows)
#     print(image_cols)

#     #data, head = nrrd.read('raw/train-sadf123bak/P0025_T1F_IM_0030_pr1_0_fan_1_mask.nrrd')
#     data, head = nrrd.read('raw/train/P0025_T1F_IM_0030_pr1_0_fan_1.nrrd')
#     f = open('ramp.png', 'wb')
#     image = data.squeeze()
#     w = png.Writer(image.shape[1], image.shape[0], greyscale=True)
#     w.write(f, image)
#     f.close()
#     exit()
#     # print(np.amax(data))
#     data1, head = nrrd.read('raw/train/P0025_T1F_IM_0030_pr1_0_fan_1_mask.nrrd')
#     data2[data2 > 254] = 1
#     print(np.amax(data2))
#     print(np.array_equal(data, data2))
#     exit()


# Preporcessing helpers

def max_image_rows():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    max_rows = 0
    for image_name in images:
        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        if max_rows < img.shape[0]:
            max_rows = img.shape[0]
    return max_rows

def max_image_cols():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)

    max_cols = 0
    for image_name in images:
        img, head = nrrd.read(os.path.join(train_data_path, image_name))
        if max_cols < img.shape[0]:
           max_cols = img.shape[1]
    return max_cols

# Main handler
if __name__ == '__main__':
    preprocess_train_data()
    cleanup_broken_data()
    create_cropped_train_data() # or create_resized_train_data()
    preprocess_test_data()  # experimental; unsued
    create_test_data()
