from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Activation, Layer, Conv2DTranspose, Dropout, Add
from keras.models import Model
from keras.activations import relu, softmax
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate, add

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


def segmentation_model(input_shape):
	#input_shape = (224, 224, 3)
	pool_size = (2,2)

	inputs = Input(shape=input_shape)

	layer1 = BatchNormalization()(inputs)
	# Encoder
	# 1
	conv1 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(layer1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv2 = Conv2D(16, (3,3),  padding='same', kernel_initializer='he_normal')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool1, mask1 = MaxPoolingWithArgmax2D(pool_size)(conv2)
	drop1 = Dropout(0.2)(pool1)

	# 2
	conv3 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(drop1)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	conv4 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(conv3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool2, mask2 = MaxPoolingWithArgmax2D(pool_size)(conv4)
	drop2 = Dropout(0.2)(pool2)

	# 3
	conv5 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(drop2)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)
	conv6 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)
	conv7 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv6)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation('relu')(conv7)
	pool3, mask3 = MaxPoolingWithArgmax2D(pool_size)(conv7)
	drop3 = Dropout(0.2)(pool3)

	# 4
	conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(drop3)
	conv8 = BatchNormalization()(conv8)
	conv8 = Activation('relu')(conv8)
	conv9 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv8)
	conv9 = BatchNormalization()(conv9)
	conv9 = Activation('relu')(conv9)
	conv10 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = BatchNormalization()(conv10)
	conv10 = Activation('relu')(conv10)
	pool4, mask4 = MaxPoolingWithArgmax2D(pool_size)(conv10)
	drop4 = Dropout(0.2)(pool4)

	# 5 
	conv11 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(drop4)
	conv11 = BatchNormalization()(conv11)
	conv11 = Activation('relu')(conv11)
	conv12 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv11)
	conv12 = BatchNormalization()(conv12)
	conv12 = Activation('relu')(conv12)
	conv13 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv12)
	conv13 = BatchNormalization()(conv13)
	conv13 = Activation('relu')(conv13)
	pool5, mask5 = MaxPoolingWithArgmax2D(pool_size)(conv13)
	drop5 = Dropout(0.2)(pool5)


	# Decoder
	# 5
	unpool5 = MaxUnpooling2D(pool_size)([drop5, mask5])
	unconv13 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unpool5)
	unconv13 = BatchNormalization()(unconv13)
	unconv13 = Activation('relu')(unconv13)
	unconv12 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv13)
	unconv12 = BatchNormalization()(unconv12)
	unconv12 = Activation('relu')(unconv12)
	unconv11 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv12)
	unconv11 = BatchNormalization()(unconv11)
	unconv11 = Activation('relu')(unconv11)
	drop6 = Dropout(0.2)(unconv11)

	# 4
	unpool4 = MaxUnpooling2D(pool_size)([drop6, mask4])
	unconv10 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unpool4)
	unconv10 = BatchNormalization()(unconv10)
	unconv10 = Activation('relu')(unconv10)
	unconv9 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv10)
	unconv9 = BatchNormalization()(unconv9)
	unconv9 = Activation('relu')(unconv9)
	unconv8 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(unconv9)
	unconv8 = BatchNormalization()(unconv8)
	unconv8 = Activation('relu')(unconv8)
	drop7 = Dropout(0.2)(unconv8)

	# 3
	unpool3 = MaxUnpooling2D(pool_size)([drop7, mask3])
	unconv7 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(unpool3)
	unconv7 = BatchNormalization()(unconv7)
	unconv7 = Activation('relu')(unconv7)
	unconv6 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(unconv7)
	unconv6 = BatchNormalization()(unconv6)
	unconv6 = Activation('relu')(unconv6)
	unconv5 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(unconv6)
	unconv5 = BatchNormalization()(unconv5)
	unconv5 = Activation('relu')(unconv5)
	drop8 = Dropout(0.2)(unconv5)

	# 2
	unpool2 = MaxUnpooling2D(pool_size)([drop8, mask2])
	unconv4 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(unpool2)
	unconv4 = BatchNormalization()(unconv4)
	unconv4 = Activation('relu')(unconv4)
	unconv3 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(unconv4)
	unconv3 = BatchNormalization()(unconv3)
	unconv3 = Activation('relu')(unconv3)
	drop9 = Dropout(0.2)(unconv3)

	# 1 
	unpool1 = MaxUnpooling2D(pool_size)([drop9, mask1])
	unconv2 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(unpool1)
	unconv2 = BatchNormalization()(unconv2)
	unconv2 = Activation('relu')(unconv2)
	unconv1 = Conv2D(1, (3,3),  padding='same', kernel_initializer='he_normal')(unconv2)
	unconv1 = BatchNormalization()(unconv1)
	unconv1 = Activation('sigmoid')(unconv1)

	model = Model(input=inputs, output=unconv1)
	return model


def segmentation_model2(input_shape):
	#input_shape = (224, 224, 3)
	pool_size = (2,2)

	inputs = Input(shape=input_shape)

	layer1 = BatchNormalization()(inputs)
	# Encoder
	# 1
	conv1 = Conv2D(32, (3,3), padding='same')(layer1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv2 = Conv2D(32, (3,3),  padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool1, mask1 = MaxPoolingWithArgmax2D(pool_size)(conv2)
	drop1 = Dropout(0.2)(pool1)

	# 2
	conv3 = Conv2D(64, (3,3), padding='same')(drop1)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	conv4 = Conv2D(64, (3,3), padding='same')(conv3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool2, mask2 = MaxPoolingWithArgmax2D(pool_size)(conv4)
	drop2 = Dropout(0.2)(pool2)

	# 3
	conv5 = Conv2D(128, (3,3), padding='same')(drop2)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)
	conv6 = Conv2D(128, (3,3), padding='same')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)
	conv7 = Conv2D(128, (3,3), padding='same')(conv6)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation('relu')(conv7)
	pool3, mask3 = MaxPoolingWithArgmax2D(pool_size)(conv7)
	drop3 = Dropout(0.2)(pool3)

	# 4
	conv8 = Conv2D(256, (3,3), padding='same')(drop3)
	conv8 = BatchNormalization()(conv8)
	conv8 = Activation('relu')(conv8)
	conv9 = Conv2D(256, (3,3), padding='same')(conv8)
	conv9 = BatchNormalization()(conv9)
	conv9 = Activation('relu')(conv9)
	conv10 = Conv2D(256, (3,3), padding='same')(conv9)
	conv10 = BatchNormalization()(conv10)
	conv10 = Activation('relu')(conv10)
	pool4, mask4 = MaxPoolingWithArgmax2D(pool_size)(conv10)
	drop4 = Dropout(0.2)(pool4)

	# 5 
	conv11 = Conv2D(256, (3,3), padding='same')(drop4)
	conv11 = BatchNormalization()(conv11)
	conv11 = Activation('relu')(conv11)
	conv12 = Conv2D(256, (3,3), padding='same')(conv11)
	conv12 = BatchNormalization()(conv12)
	conv12 = Activation('relu')(conv12)
	conv13 = Conv2D(256, (3,3), padding='same')(conv12)
	conv13 = BatchNormalization()(conv13)
	conv13 = Activation('relu')(conv13)
	pool5, mask5 = MaxPoolingWithArgmax2D(pool_size)(conv13)
	drop5 = Dropout(0.2)(pool5)


	# Decoder
	# 5
	unpool5 = MaxUnpooling2D(pool_size)([drop5, mask5])
	con1 = concatenate([unpool5, conv13])
	unconv13 = Conv2D(256, (3,3), padding='same')(con1)
	unconv13 = BatchNormalization()(unconv13)
	unconv13 = Activation('relu')(unconv13)
	unconv12 = Conv2D(256, (3,3), padding='same')(unconv13)
	unconv12 = BatchNormalization()(unconv12)
	unconv12 = Activation('relu')(unconv12)
	unconv11 = Conv2D(256, (3,3), padding='same')(unconv12)
	unconv11 = BatchNormalization()(unconv11)
	unconv11 = Activation('relu')(unconv11)
	drop6 = Dropout(0.2)(unconv11)

	# 4
	unpool4 = MaxUnpooling2D(pool_size)([drop6, mask4])
	con2 = concatenate([unpool4, conv10])
	unconv10 = Conv2D(256, (3,3), padding='same')(con2)
	unconv10 = BatchNormalization()(unconv10)
	unconv10 = Activation('relu')(unconv10)
	unconv9 = Conv2D(256, (3,3), padding='same')(unconv10)
	unconv9 = BatchNormalization()(unconv9)
	unconv9 = Activation('relu')(unconv9)
	unconv8 = Conv2D(128, (3,3), padding='same')(unconv9)
	unconv8 = BatchNormalization()(unconv8)
	unconv8 = Activation('relu')(unconv8)
	drop7 = Dropout(0.2)(unconv8)

	# 3
	unpool3 = MaxUnpooling2D(pool_size)([drop7, mask3])
	con3 = concatenate([unpool3, conv7])
	unconv7 = Conv2D(128, (3,3), padding='same')(con3)
	unconv7 = BatchNormalization()(unconv7)
	unconv7 = Activation('relu')(unconv7)
	unconv6 = Conv2D(128, (3,3), padding='same')(unconv7)
	unconv6 = BatchNormalization()(unconv6)
	unconv6 = Activation('relu')(unconv6)
	unconv5 = Conv2D(64, (3,3), padding='same')(unconv6)
	unconv5 = BatchNormalization()(unconv5)
	unconv5 = Activation('relu')(unconv5)
	drop8 = Dropout(0.2)(unconv5)

	# 2
	unpool2 = MaxUnpooling2D(pool_size)([drop8, mask2])
	con4 = concatenate([unpool2, conv4])
	unconv4 = Conv2D(64, (3,3), padding='same')(con4)
	unconv4 = BatchNormalization()(unconv4)
	unconv4 = Activation('relu')(unconv4)
	unconv3 = Conv2D(32, (3,3), padding='same')(unconv4)
	unconv3 = BatchNormalization()(unconv3)
	unconv3 = Activation('relu')(unconv3)
	drop9 = Dropout(0.2)(unconv3)

	# 1 
	unpool1 = MaxUnpooling2D(pool_size)([drop9, mask1])
	con5 = concatenate([unpool1, conv2])
	unconv2 = Conv2D(32, (3,3), padding='same')(unpool1)
	unconv2 = BatchNormalization()(unconv2)
	unconv2 = Activation('relu')(unconv2)
	unconv1 = Conv2D(1, (3,3),  padding='same')(unconv2)
	unconv1 = BatchNormalization()(unconv1)
	unconv1 = Activation('sigmoid')(unconv1)

	model = Model(input=inputs, output=unconv1)
	return model

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img_shape, n_filters = 32, dropout = 0.2, batchnorm = True):
    """Function to define the UNET Model"""
    input_img = Input(shape=input_img_shape)
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = BatchNormalization()(u6)
    u6 = Activation('relu')(u6)
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = BatchNormalization()(u7)
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = BatchNormalization()(u8)
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = BatchNormalization()(u9)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

VGG_Weights_path = "/home/beast/Desktop/inthiyaz_segmentation/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


def FCN8(nClasses, input_height=224, input_width=224):
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 7, 7, 512)

    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # <--> o = ( Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    # x = Dense(4096, activation='relu', name='fc2')(x)
    # <--> o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    # x = Dense(1000 , activation='softmax', name='predictions')(x)
    # <--> o = ( Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o)
    # assuming that the input_height = input_width = 224 as in VGG data

    vgg = Model(img_input, pool5)
    vgg.load_weights(VGG_Weights_path)  ## loading VGG weights for the encoder parts of FCN8

    n = 32
    o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=IMAGE_ORDERING)(
        conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)

    pool311 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('sigmoid'))(o)

    model = Model(img_input, o)

    return model


def segmentation_model3(input_shape):
	#input_shape = (224, 224, 3)
	pool_size = (2,2)

	inputs = Input(shape=input_shape)

	layer1 = BatchNormalization()(inputs)
	# Encoder
	# 1
	conv1 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(layer1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Activation('relu')(conv1)
	conv2 = Conv2D(16, (3,3),  padding='same', kernel_initializer='he_normal')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Activation('relu')(conv2)
	pool1, mask1 = MaxPoolingWithArgmax2D(pool_size)(conv2)
	drop1 = Dropout(0.2)(pool1)

	# 2
	conv3 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(drop1)
	conv3 = BatchNormalization()(conv3)
	conv3 = Activation('relu')(conv3)
	conv4 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(conv3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Activation('relu')(conv4)
	pool2, mask2 = MaxPoolingWithArgmax2D(pool_size)(conv4)
	drop2 = Dropout(0.2)(pool2)

	# 3
	conv5 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(drop2)
	conv5 = BatchNormalization()(conv5)
	conv5 = Activation('relu')(conv5)
	conv6 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv5)
	conv6 = BatchNormalization()(conv6)
	conv6 = Activation('relu')(conv6)
	conv7 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(conv6)
	conv7 = BatchNormalization()(conv7)
	conv7 = Activation('relu')(conv7)
	pool3, mask3 = MaxPoolingWithArgmax2D(pool_size)(conv7)
	drop3 = Dropout(0.2)(pool3)

	# 4
	conv8 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(drop3)
	conv8 = BatchNormalization()(conv8)
	conv8 = Activation('relu')(conv8)
	conv9 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv8)
	conv9 = BatchNormalization()(conv9)
	conv9 = Activation('relu')(conv9)
	conv10 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv9)
	conv10 = BatchNormalization()(conv10)
	conv10 = Activation('relu')(conv10)
	pool4, mask4 = MaxPoolingWithArgmax2D(pool_size)(conv10)
	drop4 = Dropout(0.2)(pool4)

	# 5 
	conv11 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(drop4)
	conv11 = BatchNormalization()(conv11)
	conv11 = Activation('relu')(conv11)
	conv12 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv11)
	conv12 = BatchNormalization()(conv12)
	conv12 = Activation('relu')(conv12)
	conv13 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(conv12)
	conv13 = BatchNormalization()(conv13)
	conv13 = Activation('relu')(conv13)
	pool5, mask5 = MaxPoolingWithArgmax2D(pool_size)(conv13)
	drop5 = Dropout(0.2)(pool5)


	# Decoder
	# 5
	unpool5 = MaxUnpooling2D(pool_size)([drop5, mask5])
	con1 = concatenate([unpool5, conv13])
	unconv13 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(con1)
	unconv13 = BatchNormalization()(unconv13)
	unconv13 = Activation('relu')(unconv13)
	unconv12 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv13)
	unconv12 = BatchNormalization()(unconv12)
	unconv12 = Activation('relu')(unconv12)
	unconv11 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv12)
	unconv11 = BatchNormalization()(unconv11)
	unconv11 = Activation('relu')(unconv11)
	drop6 = Dropout(0.2)(unconv11)

	# 4
	unpool4 = MaxUnpooling2D(pool_size)([drop6, mask4])
	con2 = concatenate([unpool4, conv10])
	unconv10 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(con2)
	unconv10 = BatchNormalization()(unconv10)
	unconv10 = Activation('relu')(unconv10)
	unconv9 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(unconv10)
	unconv9 = BatchNormalization()(unconv9)
	unconv9 = Activation('relu')(unconv9)
	unconv8 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(unconv9)
	unconv8 = BatchNormalization()(unconv8)
	unconv8 = Activation('relu')(unconv8)
	drop7 = Dropout(0.2)(unconv8)

	# 3
	unpool3 = MaxUnpooling2D(pool_size)([drop7, mask3])
	con3 = concatenate([unpool3, conv7])
	unconv7 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(con3)
	unconv7 = BatchNormalization()(unconv7)
	unconv7 = Activation('relu')(unconv7)
	unconv6 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(unconv7)
	unconv6 = BatchNormalization()(unconv6)
	unconv6 = Activation('relu')(unconv6)
	unconv5 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(unconv6)
	unconv5 = BatchNormalization()(unconv5)
	unconv5 = Activation('relu')(unconv5)
	drop8 = Dropout(0.2)(unconv5)

	# 2
	unpool2 = MaxUnpooling2D(pool_size)([drop8, mask2])
	con4 = concatenate([unpool2, conv4])
	unconv4 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(con4)
	unconv4 = BatchNormalization()(unconv4)
	unconv4 = Activation('relu')(unconv4)
	unconv3 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(unconv4)
	unconv3 = BatchNormalization()(unconv3)
	unconv3 = Activation('relu')(unconv3)
	drop9 = Dropout(0.2)(unconv3)

	# 1 
	unpool1 = MaxUnpooling2D(pool_size)([drop9, mask1])
	con5 = concatenate([unpool1, conv2])
	unconv2 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(unpool1)
	unconv2 = BatchNormalization()(unconv2)
	unconv2 = Activation('relu')(unconv2)
	unconv1 = Conv2D(3, (3,3),  padding='same', kernel_initializer='he_normal')(unconv2)
	unconv1 = BatchNormalization()(unconv1)
	unconv1 = Activation('softmax')(unconv1)

	model = Model(input=inputs, output=unconv1)
	return model

