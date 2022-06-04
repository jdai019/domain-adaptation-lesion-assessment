import keras.backend as K
from keras.models import Model
from keras.layers import Input, ZeroPadding2D, concatenate, add,multiply
from keras.layers.core import Activation,Dense,Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def sinnet(check_class=1,lr=1e-8):
    concat_axis = 3

    img_input = Input(shape=(224, 224, 1))
    conv1 =Conv2D(32,(3,3),padding='same',name='conv1')(img_input)
    bn1=conv1
   # bn1 = BatchNormalization(name='conv1_bn')(conv1)
    ac1 = Activation('relu', name='relu1')(bn1)
    pool1 = MaxPooling2D((2, 2), name='pool1')(ac1)

    conv2 = Conv2D(64, (3, 3), padding='same', name='conv2')(pool1)
    bn2=conv2
   # bn2 = BatchNormalization(name='conv2_bn')(conv2)
    ac2 = Activation('relu', name='relu2')(bn2)
    pool2 = MaxPooling2D((2, 2), name='pool2')(ac2)

    conv3 = Conv2D(128, (3, 3), padding='same', name='conv3')(pool2)
    bn3=conv3
   # bn3 = BatchNormalization(name='conv3_bn')(conv3)
    ac3 = Activation('relu', name='relu3')(bn3)
    pool3 = MaxPooling2D((2, 2), name='pool3')(ac3)

    conv4 = Conv2D(256, (3, 3), padding='same', name='conv4')(pool3)
    bn4=conv4
   # bn4 = BatchNormalization(name='conv4_bn')(conv4)
    ac4 = Activation('relu', name='relu4')(bn4)
    pool4 = MaxPooling2D((2, 2),name='pool4')(ac4)

    conv5 = Conv2D(256, (3, 3), padding='same', name='conv5')(pool4)
    bn5=conv5
   # bn5 = BatchNormalization(name='conv5_bn')(conv5)
    ac5 = Activation('relu', name='relu5')(bn5)

    up1=UpSampling2D((2,2))(ac5)
    merge1=concatenate([ac4,up1],axis=concat_axis,name='concat1')
    conv6 = Conv2D(128, (3, 3), padding='same', name='conv6')(merge1)
    bn6=conv6
   # bn6 = BatchNormalization(name='conv6_bn')(conv6)
    ac6 = Activation('relu', name='relu6')(bn6)

    up2 = UpSampling2D((2, 2))(ac6)
    merge2 = concatenate([ac3, up2], axis=concat_axis, name='concat2')
    conv7 = Conv2D(64, (3, 3), padding='same', name='conv7')(merge2)
    bn7=conv7
   # bn7 = BatchNormalization(name='conv7_bn')(conv7)
    ac7 = Activation('relu', name='relu7')(bn7)

    up3 = UpSampling2D((2, 2))(ac7)
    merge3 = concatenate([ac2, up3], axis=concat_axis, name='concat3')
    conv8 = Conv2D(32, (3, 3), padding='same', name='conv8')(merge3)
    bn8=conv8
   # bn8 = BatchNormalization(name='conv8_bn')(conv8)
    ac8 = Activation('relu', name='relu8')(bn8)

    up4 = UpSampling2D((2, 2))(ac8)
    merge4 = concatenate([ac1, up4], axis=concat_axis, name='concat4')
    conv9 = Conv2D(32, (3, 3), padding='same', name='conv9')(merge4)
    bn9=conv9
   # bn9 = BatchNormalization(name='conv9_bn')(conv9)
    ac9= Activation('relu', name='relu9')(bn9)
    mask=Conv2D(1, (3, 3), padding='same', name='mask',activation='sigmoid')(ac9)


    #lesion classification

    lesion=multiply([img_input,mask])
    conv11 = Conv2D(32, (3, 3), padding='same', name='conv11')(lesion)
    bn11=conv11
   # bn11 = BatchNormalization(name='conv11_bn')(conv11)
    ac11= Activation('relu', name='relu11')(bn11)
    ac11=concatenate([ac1,ac11],axis=concat_axis)
    pool11 = MaxPooling2D((2,2), name='pool11')(ac11)

    conv12 = Conv2D(64, (3, 3), padding='same', name='conv12')(pool11)
    bn12=conv12
   # bn12 = BatchNormalization(name='conv12_bn')(conv12)
    ac12 = Activation('relu', name='relu12')(bn12)
    ac12=concatenate([ac2,ac12],axis=concat_axis)
    pool12 = MaxPooling2D((2, 2), name='pool12')(ac12)

    conv13 = Conv2D(128, (3, 3), padding='same', name='conv13')(pool12)
    bn13=conv13
   # bn13 = BatchNormalization(name='conv13_bn')(conv13)
    ac13 = Activation('relu', name='relu13')(bn13)
    ac13=concatenate([ac3,ac13],axis=concat_axis)
    pool13 = MaxPooling2D((2, 2), name='pool13')(ac13)

    conv14 = Conv2D(256, (3, 3), padding='same', name='conv14')(pool13)
    bn14=conv14
   # bn14 = BatchNormalization(name='conv14_bn')(conv14)
    ac14 = Activation('relu', name='relu14')(bn14)
    ac14=concatenate([ac4,ac14],axis=concat_axis)
    pool14 = MaxPooling2D((2, 2), name='pool14')(ac14)

    conv15 = Conv2D(256, (3, 3), padding='same', name='conv15')(pool14)
    bn15=conv15
   # bn15 = BatchNormalization(name='conv15_bn')(conv15)
    ac15 = Activation('relu', name='relu15')(bn15)
    ac15=concatenate([ac5,ac15],axis=concat_axis)

    fl = Flatten(name='flatten_1')(ac15)
    dense1 = Dense(1024, activation='relu', kernel_initializer="normal", name='dense_1')(fl)
    lesion_class = Dense(2, activation='softmax', kernel_initializer="normal", name='class')(dense1)

    model = Model(img_input, [mask, lesion_class], name='sinnet')

    losses = {
        "class": 'categorical_crossentropy',
        "mask": dice_coef_loss
    }

    metrics = {
        "class": 'accuracy',
        "mask": dice_coef
    }

    loss_Weights={
        "class": check_class,
        "mask": 1.0
    }

    optim = Adam(lr=lr)
    model.compile(optimizer=optim, loss=losses,loss_weights=loss_Weights, metrics=metrics)

    return model



