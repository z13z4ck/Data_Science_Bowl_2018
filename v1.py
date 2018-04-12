# import all
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import transform
import cv2, os, sys, tqdm, random, skimage

#use optimised settings
cv2.setUseOptimized(True)

_BATCH = 128
_EPOCH = 1000
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


def loadalldata(aug = 1,size=(_IMGHEIGHT,_IMGWIDTH)):
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
                if str(train_df['img_name'].loc[i].replace(".png","")) == _HARDIMG:
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
    x_train = np.array(x_train)
    y_train = np.array(y_train)[:,:,:,np.newaxis]
    x_test = np.array(x_test)

    print('x_train.shape: {} of dtype {}'.format(x_train.shape, x_train.dtype))
    print('y_train.shape: {} of dtype {}'.format(y_train.shape, y_train.dtype))
    print('x_test.shape: {} of dtype {}'.format(x_test.shape, x_test.dtype))

    return x_train, y_train, x_test

def get_variable(name,shape):
    return tf.get_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer())

def UNet(X):
    ### Unit 1 ###
    with tf.name_scope('Unit1'):
        W1_1 =   get_variable("W1_1", [3,3,3,16] )
        Z1 = tf.nn.conv2d(X,W1_1, strides = [1,1,1,1], padding = 'SAME')
        A1 = tf.nn.relu(Z1)
        W1_2 =   get_variable("W1_2", [3,3,16,16] )
        Z2 = tf.nn.conv2d(A1,W1_2, strides = [1,1,1,1], padding = 'SAME')
        A2 = tf.nn.relu(Z2)
        P1 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 2 ###
    with tf.name_scope('Unit2'):
        W2_1 =   get_variable("W2_1", [3,3,16,32] )
        Z3 = tf.nn.conv2d(P1,W2_1, strides = [1,1,1,1], padding = 'SAME')
        A3 = tf.nn.relu(Z3)
        W2_2 =   get_variable("W2_2", [3,3,32,32] )
        Z4 = tf.nn.conv2d(A3,W2_2, strides = [1,1,1,1], padding = 'SAME')
        A4 = tf.nn.relu(Z4)
        P2 = tf.nn.max_pool(A4, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 3 ###
    with tf.name_scope('Unit3'):
        W3_1 =   get_variable("W3_1", [3,3,32,64] )
        Z5 = tf.nn.conv2d(P2,W3_1, strides = [1,1,1,1], padding = 'SAME')
        A5 = tf.nn.relu(Z5)
        W3_2 =   get_variable("W3_2", [3,3,64,64] )
        Z6 = tf.nn.conv2d(A5,W3_2, strides = [1,1,1,1], padding = 'SAME')
        A6 = tf.nn.relu(Z6)
        P3 = tf.nn.max_pool(A6, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 4 ###
    with tf.name_scope('Unit4'):
        W4_1 =   get_variable("W4_1", [3,3,64,128] )
        Z7 = tf.nn.conv2d(P3,W4_1, strides = [1,1,1,1], padding = 'SAME')
        A7 = tf.nn.relu(Z7)
        W4_2 =   get_variable("W4_2", [3,3,128,128] )
        Z8 = tf.nn.conv2d(A7,W4_2, strides = [1,1,1,1], padding = 'SAME')
        A8 = tf.nn.relu(Z8)
        P4 = tf.nn.max_pool(A8, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    ### Unit 5 ###
    with tf.name_scope('Unit5'):
        W5_1 =   get_variable("W5_1", [3,3,128,256] )
        Z9 = tf.nn.conv2d(P4,W5_1, strides = [1,1,1,1], padding = 'SAME')
        A9 = tf.nn.relu(Z9)
        W5_2 =   get_variable("W5_2", [3,3,256,256] )
        Z10 = tf.nn.conv2d(A9,W5_2, strides = [1,1,1,1], padding = 'SAME')
        A10 = tf.nn.relu(Z10)
    ### Unit 6 ###
    with tf.name_scope('Unit6'):
        W6_1 =   get_variable("W6_1", [3,3,256,128] )
        U1 = tf.layers.conv2d_transpose(A10, filters = 128, kernel_size = 2, strides = 2, padding = 'SAME')
        U1 = tf.concat([U1, A8],3)
        W6_2 =   get_variable("W6_2", [3,3,128,128] )
        Z11 = tf.nn.conv2d(U1,W6_1, strides = [1,1,1,1], padding = 'SAME')
        A11 = tf.nn.relu(Z11)
        Z12 = tf.nn.conv2d(A11,W6_2, strides = [1,1,1,1], padding = 'SAME')
        A12 = tf.nn.relu(Z12)
    ### Unit 7 ###
    with tf.name_scope('Unit7'):
        W7_1 =   get_variable("W7_1", [3,3,128,64] )
        U2 = tf.layers.conv2d_transpose(A12, filters = 64, kernel_size = 2, strides = 2, padding = 'SAME')
        U2 = tf.concat([U2, A6],3)
        Z13 = tf.nn.conv2d(U2,W7_1, strides = [1,1,1,1], padding = 'SAME')
        A13 = tf.nn.relu(Z13)
        W7_2 =   get_variable("W7_2", [3,3,64,64] )
        Z14 = tf.nn.conv2d(A13,W7_2, strides = [1,1,1,1], padding = 'SAME')
        A14 = tf.nn.relu(Z14)
    ### Unit 8 ###
    with tf.name_scope('Unit8'):
        W8_1 =   get_variable("W8_1", [3,3,64,32] )
        U3 = tf.layers.conv2d_transpose(A14, filters = 32, kernel_size = 2, strides = 2, padding = 'SAME')
        U3 = tf.concat([U3, A4],3)
        Z15 = tf.nn.conv2d(U3,W8_1, strides = [1,1,1,1], padding = 'SAME')
        A15 = tf.nn.relu(Z15)
        W8_2 =   get_variable("W8_2", [3,3,32,32] )
        Z16 = tf.nn.conv2d(A15,W8_2, strides = [1,1,1,1], padding = 'SAME')
        A16 = tf.nn.relu(Z16)
    ### Unit 9 ###
    with tf.name_scope('Unit9'):
        W9_1 =   get_variable("W9_1", [3,3,32,16] )
        U4 = tf.layers.conv2d_transpose(A16, filters = 16, kernel_size = 2, strides = 2, padding = 'SAME')
        U4 = tf.concat([U4, A2],3)
        Z17 = tf.nn.conv2d(U4,W9_1, strides = [1,1,1,1], padding = 'SAME')
        A17 = tf.nn.relu(Z17)
        W9_2 =   get_variable("W9_2", [3,3,16,16] )
        Z18 = tf.nn.conv2d(A17,W9_2, strides = [1,1,1,1], padding = 'SAME')
        A18 = tf.nn.relu(Z18)
    ### Unit 10 ###
    with tf.name_scope('out_put'):
        W10 =    get_variable("W10", [1,1,16,1] )
        Z19 = tf.nn.conv2d(A18,W10, strides = [1,1,1,1], padding = 'SAME')
        A19 = tf.nn.sigmoid(Z19)
        Y_pred = A19
    return Y_pred

def loss_function(y_pred, y_true):
    cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true,y_pred))
    return cost

def mean_iou(y_pred,y_true):
    y_pred_ = tf.to_int64(y_pred > 0.5)
    y_true_ = tf.to_int64(y_true > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true_, y_pred_, 2)
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

# build the graph as a dictionary
def build_graph():
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            with tf.name_scope('input'):
                x_ = tf.placeholder(tf.float32, shape=(None,_IMGHEIGHT, _IMGWIDTH, _IMGCOLOR))
                y_ = tf.placeholder(tf.float32, shape=(None,_IMGHEIGHT, _IMGWIDTH, 1))
            y_pred = UNet(x_)
            with tf.name_scope('loss'):
                loss = loss_function(y_pred,y_)
        with tf.device("/cpu:0"):
            with tf.name_scope("metrics"):
                iou = mean_iou(y_pred,y_)
    return g,x_, y_, y_pred, loss, iou

def next_batch(num, x_train, y_train):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(x_train))
    np.random.shuffle(idx)
    idx = idx[:num]
    x_shuffle = [x_train[ i] for i in idx]
    y_shuffle = [y_train[ i] for i in idx]
    return np.asarray(x_shuffle), np.asarray(y_shuffle)

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
dataset = len(x_train)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
graph, _x, _y, y_pred, loss, iou = build_graph()
with tf.Session(graph=graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('_logs/' + "BIO", graph)
    writer.add_graph(sess.graph)
    sys.stdout.flush()
    _loss = 0
    _itter = 0
    _iou = 0
    _summary = 0
    epochbar = tqdm.tqdm(range(0,_EPOCH), total=_EPOCH, unit='epoch')
    for epoch_step in epochbar:
        epochbar.set_description("Loss : {} | Itter : {} | IOU : {} ".format(_loss,_itter, _iou))
        for batch_step in range(0,round(dataset/_BATCH)):
            x_train,y_train = next_batch(_BATCH,x_train,y_train)
            feed = {_x: x_train, _y: y_train}
            _ypred, _loss, _iou  = (sess.run([y_pred,loss,iou], feed_dict = feed ))
            _itter += 1
    for i, filename in tqdm.tqdm(enumerate(test_df['image_path']), total=len(train_df), unit='images'):
        testfeed = {_x: x_test[i]}
        _finalpred = sess.run(y_pred, feed_dict=testfeed)
        if not os.path.exists(os.path.join(_TESTPATH, str(test_df['img_name'].loc[i]).replace(".png", ""), "result")): os.makedirs(os.path.join(_TESTPATH, str(test_df['img_name'].loc[i]).replace(".png", ""), "result"))
        plt.imsave(fname=os.path.join(_TESTPATH, str(test_df['img_name'].loc[i]).replace(".png", ""), "result",str(test_df['img_name'].loc[i])),arr=_finalpred)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

