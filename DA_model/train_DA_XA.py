import numpy as np

#from denseunet import DenseUNet
from sinnet_nobn_share_bk import sinnet_share
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score,confusion_matrix
from keras.models import load_model
import random
import time
import keras.backend as K
import gc
from keras.utils.vis_utils import plot_model
from train_seperate import read_samples, dice_coef, dice_coef_loss

def CORAL(source, target):
    d = K.int_shape(source)[1]

    xm = K.mean(source,axis=0,keepdims=True) - source
    xc = K.transpose(xm) @ xm

    xmt = K.mean(target,axis=0,keepdims=True) - target

    xct = K.transpose(xmt) @ xmt

    loss = K.mean(K.batch_dot((xc - xct),(xc - xct)),axis=0, keepdims=True)
    loss = loss/(4*d*d)

    return loss

def train_and_predict(src_dir,src_mask_dir,tgt_dir,tgt_mask_dir,tgt_dir2,tgt_mask_dir2,tgt_dir3,tgt_mask_dir3,start,stop,sequence,check_class,epoch,weightfile,lr,method):
    
    filepath='{0}_DA_XA/{1}_sinnet_{0}'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)))

    if check_class == 1:
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='auto',save_best_only='True')

    elif check_class == 0:
        checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_mask_dice_coef', mode='max', save_best_only='True')
    callback_lists = [checkpoint]
    model=sinnet_share(weightfile,lr)

    if os.path.exists(filepath):
        model.load_weights(filepath)

    train_images,masks,classes,pt_list,pf_list,total_list = read_samples(src_dir,src_mask_dir,method)
    train_images_,masks_,classes_,pt_list_,pf_list_,total_list_ = read_samples(tgt_dir,tgt_mask_dir,method)

    ################

    val_list = pt_list[int(start*len(pt_list)):int(stop*len(pt_list))] + pf_list[int(start*len(pf_list)):int(stop*len(pf_list))]
    train_list = list(set(total_list).difference(set(val_list)))
    val_list_ = pt_list_[int(start*len(pt_list_)):int(stop*len(pt_list_))] + pf_list_[int(start*len(pf_list_)):int(stop*len(pf_list_))]

    if len(total_list_) > len(train_list):
        train_list_ = total_list_[0:len(train_list)]
    else:
        num = len(train_list)//len(total_list_)
        train_list_ = []
        for i in range(num):
            train_list_ = train_list_ + total_list_
        num1 = len(train_list) - len(train_list_)
        train_list_ = train_list_ + train_list_[0:num1]
    assert len(train_list_) == len(train_list)

    ###############

    model.fit(
        [np.array([train_images[i] for i in train_list]).reshape(-1,224,224,1),
        np.array([train_images_[i] for i in train_list_]).reshape(-1,224,224,1)],
    {"class" : np.array([classes[i] for i in train_list]).reshape(-1,2),
     "mask" : np.array([masks[i] for i in train_list]).reshape(-1,224,224,1),
     "coral": np.array([classes[i] for i in train_list]).reshape(-1,2)
     },
     epochs = epoch,
     batch_size = 2,
     validation_data =
     (
         [np.array([train_images_[i] for i in total_list_]).reshape(-1,224,224,1),
         np.array([train_images_[i] for i in total_list_]).reshape(-1,224,224,1)],
     {"class" : np.array([classes_[i] for i in total_list_]).reshape(-1,2), 
     "mask" : np.array([masks_[i] for i in total_list_] ).reshape(-1,224,224,1),
     "coral": np.array([classes_[i] for i in total_list_]).reshape(-1,2)
     }
     ),
     shuffle = True,
     callbacks = callback_lists)

    ###############

    model = load_model(
        '{0}_DA_XA/{1}_sinnet_{0}'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))), 
    custom_objects={'CORAL': CORAL},
    compile=False)
    ###############

    model.save_weights('./{0}_DA_XA/{1}_sinnet_{0}_weights.h5'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))))

    ############### PROSTATEX ####################

    aa = np.array([train_images[i] for i in val_list]).reshape(-1,224,224,1)
    predictions_prostatex = model.predict([aa,aa])
    label_prostatex = np.array([classes[i] for i in val_list]).reshape(-1,2)

    auc = roc_auc_score(label_prostatex[:,0],predictions_prostatex[1][:,0])
    print('prostatex_auc:',auc)

    conf_mat = confusion_matrix(
        np.argmax(np.array([classes[i] for i in val_list]).reshape(-1,2),axis=1),
    np.argmax(predictions_prostatex[1],axis=1))
    print('prostatex_conf_mat:\n',conf_mat)

    np.savetxt('{0}_DA_XA/px_{1}_label.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))), label_prostatex)
    np.savetxt('{0}_DA_XA/px_{1}_predictions.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))), predictions_prostatex[1])

    save_para = []
    print('sensitivity:', conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),'\n')
    print('specificity:', conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),'\n')
    print('acc:', (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0]),'\n')

    save_para.append(
        [conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),
        conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),
    (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0])])

    ################ LOCAL A #####################
    aa = np.array([train_images_[i] for i in total_list_]).reshape(-1,224,224,1)
    predictions_locala = model.predict([aa,aa])
    label_locala = np.array([classes_[i] for i in total_list_]).reshape(-1,2)
    auc = roc_auc_score(label_locala[:,0],predictions_locala[1][:,0])
    print('locala_auc:',auc)

    conf_mat = confusion_matrix(np.argmax(np.array([classes_[i] for i in total_list_]).reshape(-1,2),axis=1),np.argmax(predictions_locala[1],axis=1))
    print('locala_conf_mat:\n',conf_mat)

    np.savetxt('{0}_DA_XA/la_{1}_label.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))), label_locala)
    np.savetxt('{0}_DA_XA/la_{1}_predictions.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))), predictions_locala[1])

    print('sensitivity:', conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),'\n')
    print('specificity:', conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),'\n')
    print('acc:', (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0]),'\n')

    save_para.append([conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),
    (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0])])
    print("=========================================================================================================================")

   #####################

    model.save_weights('{0}_DA_XA/{1}_sinnet_{0}_weights.h5'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10))))

    del model
    K.clear_session()
    gc.collect()

    return start,stop


if __name__ == '__main__':

    ###############

    start = 0.6
    stop = 0.8

    check_class = 1
    epoch = 50
    method = 'scale'
    
    lr = 1e-5

    sequence = 'T2'
    print(lr,sequence)
    weightfile = "./{0}_X_A_backbone/{1}_sinnet_{0}_weights.h5".format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)))
    
    src_dir = './data/only_Xdata/T2/images'
    src_mask_dir = './data/only_Xdata/T2/labels'
    tgt_dir = './data/only_Adata/T2/images'
    tgt_mask_dir = './data/only_Adata/T2/labels'
    tgt_dir2 = './data/only_Bdata/T2/images'
    tgt_mask_dir2 = './data/only_Bdata/T2/labels'
    tgt_dir3 = './data/only_Cdata/T2/images'
    tgt_mask_dir3 = './data/only_Cdata/T2/labels'
    
    train_and_predict(src_dir,src_mask_dir,tgt_dir,tgt_mask_dir,tgt_dir2,tgt_mask_dir2,tgt_dir3,tgt_mask_dir3,start,stop,sequence,check_class,epoch,weightfile,lr,method)

   