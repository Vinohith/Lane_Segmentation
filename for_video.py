import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import cv2
from scipy.misc import imresize
from keras.models import load_model
from keras import backend as K
import time
import seaborn as sns
from keras.layers import Layer
import pickle
import matplotlib.pyplot as plt
import pickle


# coustom layers for segnet maxpooling
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == 'tensorflow':
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                    inputs,
                    ksize=ksize,
                    strides=strides,
                    padding=padding)
        else:
            errmsg = '{} backend is not supported for layer {}'.format(
                    K.backend(), type(self).__name__)
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        input_shape[0],
                        input_shape[1]*self.size[0],
                        input_shape[2]*self.size[1],
                        input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                    [[input_shape[0]], [1], [1], [1]],
                    axis=0)
            batch_range = K.reshape(
                    K.tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret
          
    def compute_output_shape(self, input_shape):
            mask_shape = input_shape[1]
            return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
)


# load the model
model = load_model('/mnt/data/MVD_research_samples/final-smaller(large_checkpoint).h5', custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D': MaxUnpooling2D})
max_seq_len = 1
x_train=[]
frames=[]
predict=[]
cap=cv2.VideoCapture("/home/beast/Desktop/inthiyaz_segmentation/Video1.mp4")
#cap=cv2.VideoCapture("/mnt/data/Black_and_white_lane_segmentation/WIN_20190627_19_21_17_Pro.mp4")



def give_color_to_seg_img(seg, n_classes):
    '''
    seg : (input_width,input_height,3)
    '''

    if len(seg.shape) == 3:
        seg = seg[:, :, 0]
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


while (cap.isOpened()):
    ret, frame1 = cap.read()

    #frame1 = cv2.imread("/mnt/data/datasets_extracted/idd/test/000455_leftImg8bit.png")
    frame1 = np.float32(cv2.resize(frame1, (512, 512)))
    cv2.imshow('b',frame1.astype(np.uint8))
    predict = np.array(frame1)
    print(predict.shape)
    predict = np.expand_dims(predict, axis=0)
    start_time = time.time()

    y_pred = model.predict(predict)

    print("FPS: ", 1.0 / (time.time() - start_time))

    y_predi = np.argmax(y_pred, axis=3)
    x=give_color_to_seg_img(y_predi[0], 3)
    x=cv2.resize(x,dsize=(512,512))
    cv2.imshow('a',x )

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
