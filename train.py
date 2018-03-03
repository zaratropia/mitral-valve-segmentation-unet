from __future__ import print_function

import Image
import png
import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import nrrd
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.callbacks import TensorBoard
tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

from data import load_train_data, load_test_data

from vis.visualization import visualize_activation, visualize_saliency, visualize_cam, overlay as vis_overlay
from vis.utils import utils
import matplotlib.pyplot as plt


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 96
img_cols = 96

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv8')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same', name='last_conv')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='dense')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('*'*30)
    print('Loading and preprocessing train data...')
    print('*'*30)

    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32') # TODO: eval float32
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    print('*'*30)
    print("Max mask value:" + str(np.amax(imgs_mask_train))
    print("Min mask value:" + str(np.amin(imgs_mask_train))

    print('*'*30)
    print('Creating and compiling model...')
    print('*'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('*'*30)
    print('Fitting model...')
    print('*'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=20, verbose=1, shuffle=True, # was True
              validation_split=0.2,
              callbacks=[model_checkpoint, tb_callback])

    print('*'*30)
    print('Loading and preprocessing test data...')
    print('*'*30)
    imgs_test, imgs_id_test, imgs_header_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('*'*30)
    print('Loading saved weights...')
    print('*'*30)
    model.load_weights('weights.h5')

    print('*'*30)
    print('Predicting masks on test data...')
    print('*'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    # This is for feature map visualization
    # filter_idx = None
    # layers = [['conv1', (0., 255.)], ['conv2', (0., 255.)], ['conv3', (0., 255.)], ['conv4', (0., 255.)], ['conv5', (0., 255.)],
    #          ['conv6', (0., 255.)], ['conv7', (0., 255.)], ['conv8', (0., 255.)], ['last_conv', (0., 255.)], ['dense', (0., 255.)]]
    # layers = []
    # for layer_name, value_range in layers:
    #     filename = 'activation' + layer_name
    #     if layer_name == 'dense':
    #         array2 = img[..., 0]
    #         layer_idx = utils.find_layer_idx(model, layer_name)
    #         penultimate_idx = utils.find_layer_idx(model, 'last_conv')
    #         img = visualize_cam(model, layer_idx, filter_indices=filter_idx, seed_input=imgs_test[0], penultimate_layer_idx=penultimate_idx)
    #         plt.imshow(img[..., 0])
    #         plt.savefig(os.path.join('visualize', filename + '_output.png'), bbox_inches='tight', pad_inches=0.0, dpi=400)
    #         array1 = img[..., 0]
    #         img_overlay = vis_overlay(array1, array2, alpha=0.5)
    #         plt.imshow(img_overlay)
    #         plt.savefig(os.path.join('visualize', filename + '_overlay.png'), bbox_inches='tight', pad_inches=0.0, dpi=400)

    #     layer_idx = utils.find_layer_idx(model, layer_name)
    #     img = visualize_activation(model, layer_idx, filter_indices=filter_idx, input_range=value_range, seed_input=imgs_test[0]) # , , tv_weight=0., lp_norm_weight=0.
    #     plt.imshow(img[..., 0])
    #     plt.savefig(os.path.join('visualize', filename + '.png'), bbox_inches='tight', pad_inches=0.0, dpi=400)


    # Save png preds for manual data analysis
    for image, test_image, image_id, image_head in zip(imgs_mask_test, imgs_test, imgs_id_test, imgs_header_test):
        if image.max() != 0: # save only masks which are not empty
            background = (test_image[:, :, 0] * 255.).astype(np.uint8)
            overlay = (image[:, :, 0] * 255.).astype(np.uint8)

            merged_image = Image.blend(Image.fromarray(background, 'L'), Image.fromarray(overlay, 'L'), 0.5)
            imsave(os.path.join(pred_dir, str(image_head["custom_name"]) + '_maskpred.png'), merged_image)

            f = open(os.path.join(pred_dir, str(image_head["custom_name"]) + '_pred.png'), 'wb')
            w = png.Writer(overlay.shape[1], overlay.shape[0], greyscale=True)
            w.write(f, overlay)
            f.close()

    # Save nrrd preds
    for image, test_image, image_id, image_head in zip(imgs_mask_test, imgs_test, imgs_id_test, imgs_header_test):
        if image.max() != 0: # save only masks which are not empty

            image = (image[:, :, 0] * 255.).astype(np.uint8)
            test_image = (test_image[:, :, 0] * 255.)

            # get original image dimensions from stored header
            rows, cols = image_head["sizes"][0], image_head["sizes"][1]
            # save mask prediction
            nrrd.write(os.path.join(pred_dir, str(image_head["custom_name"]) + '_mask_pred.nrrd'),
                resize(image, (rows , cols, 1), preserve_range=True), image_head)
            # save corresponding test image
            nrrd.write(os.path.join(pred_dir, str(image_head["custom_name"]) + '_test.nrrd'),
                resize(test_image, (rows , cols, 1), preserve_range=True), image_head)


if __name__ == '__main__':
    train_and_predict()


