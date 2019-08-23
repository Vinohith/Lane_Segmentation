from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, Adadelta
from model_train2 import segmentation_model, segmentation_model2, get_unet, FCN8, segmentation_model3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K

PATH = '/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/'

def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img = img[:, : , 0]
    x = 0
    for c in [13,24]:
        seg_labels[: , : , x ] = (img == c ).astype(int)
        x = x+1
    seg_labels[: , : , 2 ] = cv2.bitwise_or(seg_labels[: , : , 0 ]*255, seg_labels[: , : , 1 ]*255, None)
    seg_labels[: , : , 2 ] = ~np.array(seg_labels[:, :, 2], np.uint8)/255
    return seg_labels

def generator(path_frames, path_masks, batch_size):
    j = 0
    w = 512
    h = 512
    directory_img = sorted(os.listdir(path_frames))
    directory_mask = sorted(os.listdir(path_masks))
    while(1):
        images = np.zeros((batch_size, w, h, 3))
        masks = np.zeros((batch_size, w, h, 3))
        for i in range(j, j+batch_size):
            img = cv2.imread(path_frames+'/'+directory_img[i])
            img = cv2.resize(img, (w, h))
            img = img.reshape(w, h, 3)            
            images[i-j] = img
            mask = getSegmentationArr(path_masks+'/'+directory_mask[i], nClasses=3, width=w, height=h)
            masks[i-j] = mask
        j = j+batch_size
        if (j+batch_size>=len(os.listdir(path_frames))):
            j=0
        yield images, masks

#mask = getSegmentationArr('/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/training/instances/_0P04ZWQtMtPMwx3lgLdWA.png', 3, 512, 512)

model = segmentation_model3(input_shape = (512, 512, 3))
model.summary()

NO_OF_TRAINING_IMAGES = len(os.listdir(PATH+'training/images'))
NO_OF_VAL_IMAGES = len(os.listdir(PATH+'validation/images'))

NO_OF_EPOCHS = 50
BATCH_SIZE = 8


n_classes = 3


filepath="final-smaller(large_checkpoint).h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

model.compile(optimizer=adadelta, loss='categorical_crossentropy',metrics=['accuracy'])

#/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/training/images
#/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/training/instances
#/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/validation/images
#/mnt/data/datasets_extracted/mapillary-vistas-dataset_public_v1.1/validation/instances

model.fit_generator(generator(PATH+'training/images', PATH+'training/instances', BATCH_SIZE), epochs=NO_OF_EPOCHS,
               steps_per_epoch=(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
               validation_data=generator(PATH+'validation/images', PATH+'validation/instances', BATCH_SIZE),
               validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
               callbacks=callbacks_list)
model.save('final-smaller(large_3_class_512).h5')
