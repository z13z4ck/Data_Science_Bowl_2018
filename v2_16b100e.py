# import all
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform
from skimage import morphology
import cv2, os, sys, tqdm, random, skimage, datetime
from sklearn.model_selection import train_test_split
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
#use optimised settings
cv2.setUseOptimized(True)

num_gpus = 1

_BATCH = 16
_EPOCH = 5000
_HARDIMG = 'f952cc65376009cfad8249e53b9b2c0daaa3553e897096337d143c625c2df886'
#set scale train images
_IMGHEIGHT = 256
_IMGWIDTH = 256
_IMGCOLOR = 3
_IMGTYPE = '.png'
#getting workspace path
_WORKPATH = os.getcwd()
_TESTPATH = os.path.join(_WORKPATH,"stage1_test")
_TRAINPATH = os.path.join(_WORKPATH,"stage1_train")

# Set seed values
seed = 42
random.seed = seed
np.random.seed(seed=seed)

print("Workspace : {}".format(_WORKPATH))
print("Train : {}".format(_TRAINPATH))
print("Test : {}".format(_TESTPATH))

def loadimg(path, color=cv2.IMREAD_COLOR, size=None):
    img = cv2.imread(path, color)
    if size:
        img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    return img

def loadmask(dirs, size=None):
    for i,path in enumerate(next(os.walk(dirs))[2]):
        _maskpath = os.path.join(dirs, path)
        _masktmp = loadimg(_maskpath, cv2.IMREAD_GRAYSCALE, size)
        #stacking mask image
        if not i: mask = _masktmp
        else: mask = np.maximum(mask, _masktmp)
    return mask

def createdt(path):
    tmp = []
    for i, _dirpath in enumerate(os.listdir(path)):
        _imgdir = os.path.join(path,_dirpath,"images")
        _maskdir = None
        if "masks" in os.listdir(os.path.join(path,_dirpath)):
            _maskdir = os.path.join(path,_dirpath,"masks")
            _nummasks = len(os.listdir(_maskdir))
        _imgname = os.listdir(_imgdir)[0] if len(os.listdir(_imgdir)) < 2 else "Multiple Image Source"
        _imgpath = os.path.join(_imgdir,_imgname)
        _imgshape = loadimg(_imgpath).shape
        tmp.append([i, _imgname, _imgshape[0], _imgshape[1], _imgshape[0]/_imgshape[1], _imgshape[2], _nummasks, _imgpath, _maskdir] ) if _maskdir else tmp.append([i, _imgname, _imgshape[0], _imgshape[1], _imgshape[0]/_imgshape[1], _imgshape[2], _imgpath])

    dt_df = pd.DataFrame(tmp, columns= ['img_id', 'img_name', 'img_height', 'img_width', 'img_ratio', 'num_channels', 'image_path']) if len(tmp[0]) < 8 else pd.DataFrame(tmp, columns= ['img_id', 'img_name', 'img_height', 'img_width', 'img_ratio', 'num_channels', 'num_masks', 'image_path', 'mask_dir'])
    return dt_df



def augdata(img, mask, resize_rate =0.85,angle = 30):
    flip = random.randint(0, 1)
    size = img.shape[0]
    rsize = random.randint(np.floor(resize_rate*size),size)
    w_s = random.randint(0,size - rsize)
    h_s = random.randint(0,size - rsize)
    sh = random.random()/2-0.25
    rotate_angel = random.random()/180*np.pi*angle
    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=sh,rotation=rotate_angel)
    # Apply transform to image data
    img = transform.warp(img, inverse_map=afine_tf,mode='constant')
    mask = transform.warp(mask, inverse_map=afine_tf,mode='constant')
    # Randomly corpping image frame
    img = img[w_s:w_s+size,h_s:h_s+size,:]
    mask = mask[w_s:w_s+size,h_s:h_s+size]
    # Ramdomly flip frame
    if flip:
        img = img[:,::-1,:]
        mask = mask[:,::-1]
    img = transform.resize(img,(256,256),mode='edge')
    mask = transform.resize(mask,(256,256),mode='edge')
    return img, mask


def loadalldata(aug = 2, plotsample = False,size=(_IMGHEIGHT,_IMGWIDTH)):
    x_train, y_train, x_test = [], [], []

    # Read and resize train images/masks.
    print('Loading and resizing train images and masks ...')
    if aug:
        print('Image Augmentation is Enabled')
        print('Parsing train images and masks ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(train_df['image_path']), total=len(train_df), unit='images'):
        img = loadimg(train_df['image_path'].loc[i], size=size)
        mask = loadmask(train_df['mask_dir'].loc[i], size=size)
        x_train.append(img)
        y_train.append(mask)
        if aug:
            for j in range(0, int(aug)):
                augimg, augmask = augdata(img, mask)
                if not os.path.exists(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""),"augs")): os.makedirs(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""),"augs"))
                if not os.path.exists(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks")): os.makedirs(os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks"))
                if str(train_df['img_name'].loc[i].replace(".png","")) == _HARDIMG and plotsample:
                    print('Checking hardest image augmentation')
                    plt.subplot(221)
                    plt.imshow(img)
                    plt.subplot(222)
                    plt.imshow(mask)
                    plt.subplot(223)
                    plt.imshow(augimg)
                    plt.subplot(224)
                    plt.imshow(augmask)

                plt.imsave(fname= os.path.join(_TRAINPATH,str(train_df['img_name'].loc[i]).replace(".png",""),"augs", str(train_df['img_name'].loc[i]).replace(".png","_{}.png".format(j))), arr = augimg)
                plt.imsave(fname=os.path.join(_TRAINPATH, str(train_df['img_name'].loc[i]).replace(".png",""), "augs_masks", str(train_df['img_name'].loc[i]).replace(".png","_{}.png".format(j))), arr = augmask)
                x_train.append(augimg)
                y_train.append(augmask)


    # Read and resize test images.
    print('Loading and resizing test images ...')
    sys.stdout.flush()
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(test_df), unit='images'):
        img = loadimg(test_df['image_path'].loc[i], size=size)
        x_test.append(img)

    # Transform lists into 4-dim numpy arrays.
    print ("Train Images : {} and expected : {}".format(len(x_train),len(train_df)*(int(aug)+1) if aug else len(train_df)))
    x_train = np.array(x_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.bool)[:,:,:,np.newaxis]
    x_test = np.array(x_test)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, y_train.dtype))
    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))

    return x_train, y_train, x_test


#sds
train_df = createdt(_TRAINPATH)
test_df = createdt(_TESTPATH)
print('train_df:')
print(train_df.describe())
print('')
print('test_df:')
print(test_df.describe())

df = pd.DataFrame([[x] for x in zip(train_df['img_height'], train_df['img_width'])])
print (df[0].value_counts())

x_train, y_train, x_test = loadalldata()


def get_train_test_augmented(X_data=x_train, Y_data=y_train, validation_split=0.25, batch_size=32, seed=seed):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1 - validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed)

    # Image data generator distortion options
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')  # use 'constant'??

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Test data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)

    return train_generator, test_generator

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda

def Unet(imgheight, imgwidth):
    inputs = Input((imgheight, imgwidth, 3))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

import keras
from keras import backend as K
from keras.utils import multi_gpu_model, plot_model

# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        K.get_session().run(tf.global_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# Set some model compile parameters
optimizer = 'adam'
loss      = bce_dice_loss
metrics   = [mean_iou]

# Compile our model
model = Unet(_IMGHEIGHT,_IMGWIDTH)
model.summary()

# For more GPUs
if num_gpus > 1:
    model = multi_gpu_model(model, gpus=num_gpus)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

if not os.path.exists(os.path.join(_WORKPATH, "stage1_model","checkpoint")): os.makedirs(os.path.join(_WORKPATH, "stage1_model","checkpoint"))
Checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(_WORKPATH,"stage1_model","checkpoint","weights_{epoch:02d}-{val_loss:.2f}.hdf5"),save_weights_only=True)
train_generator, test_generator = get_train_test_augmented(X_data=x_train, Y_data=y_train, validation_split=0.11, batch_size=_BATCH)
model.fit_generator(train_generator, validation_data=test_generator, validation_steps=_BATCH/2, steps_per_epoch=len(x_train)/(_BATCH*2), epochs=_EPOCH, callbacks=[Checkpoint])

if num_gpus > 1:
    #Refer to https://stackoverflow.com/questions/41342098/keras-load-checkpoint-weights-hdf5-generated-by-multiple-gpus
    #model.summary()
    model_out = model.layers[-2]  #get second last layer in multi_gpu_model i.e. model.get_layer('model_1')
else:
    model_out = model
model_out.save_weights(filepath=os.path.join(_WORKPATH,"stage1_model","final-model-weights{}_{}.hdf5".format(_EPOCH,_BATCH)))
