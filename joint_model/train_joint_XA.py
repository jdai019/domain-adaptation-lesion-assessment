import numpy as np
#from denseunet import DenseUNet
from sinnet_nobn import sinnet
import sinnet_nobn
import os
from PIL import Image
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score,confusion_matrix
from keras.models import load_model
import random
import time
import keras.backend as K
import gc


#-*-coding:utf-8-*-
import os
import shutil
import SimpleITK as sitk
import warnings
import glob
import numpy as np
from nipype.interfaces.ants import N4BiasFieldCorrection

from PIL import ImageFilter

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def read_samples(image_dir,mask_dir,method):

    index=0
    true_num=0
    false_num=0 

    train_images = []
    masks = []
    classes = []

    pt_list = []
    pf_list = []

    ht_list = []
    hf_list = []

    total_list = []

    images=os.listdir(image_dir)
    images.sort()

    for image_name in images:
        if 'TRUE' in image_name:
            if 'ProstateX' in image_name:
                for i in range(10):
                    pt_list.append(index+i)
                    total_list.append(index+i)
            else:
                for i in range(1):
                    ht_list.append(index+i)
                    total_list.append(index+i)
        if 'FALSE' in image_name:
            if 'ProstateX' in image_name:
                for i in range(3):
                    pf_list.append(index+i)
                    total_list.append(index+i)
            else:
                for i in range(2):
                    hf_list.append(index+i)
                    total_list.append(index+i)

        lesion_class=image_name.split('.')[0].split('_')[-1]
        if lesion_class=='TRUE':

            if 'ProstateX' in image_name:
                true_num=true_num+10
                index+=10
                angles=[-12,-9,-6,-3,0,3,6,9,12,15]
                for i in range(10): 
                    classes.append(np.array([1,0]))
            else:
                true_num=true_num+1
                index+=1
                angles=[0]
                for i in range(1): 
                    classes.append(np.array([1,0]))
        if lesion_class=='FALSE':
            if 'ProstateX' in image_name:
                false_num=false_num+3
                index+=3
                angles=[-3,0,3]
                for i in range(3):
                    classes.append(np.array([0,1]))
            else:
                false_num = false_num + 2
                index += 2
                angles=[0,3]
                for i in range(2):
                    classes.append(np.array([0,1]))
        for angle in angles:
            image_path=os.path.join(image_dir,image_name)
            image=Image.open(image_path).convert('L')
            image=image.rotate(angle)
            image=image.resize((224,224),Image.ANTIALIAS)
            image_array=np.array(image)

            if method == 'scale':
                assert 'BFC' not in method
                image_array=(image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))

            elif method == 'standardization': 
                assert 'BFC' not in method      
                mu = np.mean(image_array, axis=0)
                sigma = np.std(image_array, axis=0)
                #print(mu,sigma)
                image_array = (image_array - mu) / (sigma+1e-7)

                #image_array=(image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))

            elif method == 'scale_BFC':
                image_array = (image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))

            elif method == 'standardization_BFC':
                
                mu = np.mean(image_array, axis=0)
                sigma = np.std(image_array, axis=0)
                image_array = (image_array - mu) / (sigma+1e-7)
                #print(mu,sigma)

                #image_array = (image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))
                
            elif method == 'scale_NF':
                image = Image.fromarray(image_array)
                image = image.filter(ImageFilter.GaussianBlur(radius=1))
                image_array=np.array(image)
                
                image_array = (image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))

            elif method == 'standardization_NF':
                
                image = Image.fromarray(image_array)
                image = image.filter(ImageFilter.GaussianBlur(radius=1))
                image_array=np.array(image)
                
                mu = np.mean(image_array, axis=0)
                sigma = np.std(image_array, axis=0)
                image_array = (image_array - mu) / (sigma+1e-7)
                #print(mu,sigma)

                #image_array = (image_array-np.amin(image_array))/((np.amax(image_array))-np.amin(image_array))

            train_images.append(image_array)
    
            mask_path=os.path.join(mask_dir,image_name)
            mask=Image.open(mask_path).convert('L')
            
            if 'ProstateX' not in mask_path:
                if 'T2' not in mask_path:
                    mask_array=np.array(mask)
                    mask_image = mask_array.transpose((1, 0))
                    mask = Image.fromarray(mask_image)
            
            mask=mask.rotate(angle)
            
            mask=mask.resize((224,224))
            mask_array=np.array(mask)/255
            masks.append(mask_array)
    print("true_num,false_num",true_num,false_num)

    return train_images,masks,classes,pt_list,pf_list,ht_list,hf_list,total_list



def train_and_predict(image_dir,mask_dir,start,stop,sequence,check_class,epoch,lr,method):
    filepath ='{0}_A_joint_{2}/{1}_sinnet_{0}'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method)
    filepath_exist = '{0}_A_joint_{2}/{1}_sinnet_{0}'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method)

    ###############
    if check_class == 1:
        checkpoint = ModelCheckpoint(
            filepath = filepath, 
            monitor='val_class_loss', 
            mode='auto',
            save_best_only='True')
        print('##################train class loss##################t')
    elif check_class == 0:
        #checkpoint = ModelCheckpoint(filepath='ADC/2_4_sinnet_ADC', monitor='val_mask_dice_coef', mode='max', save_best_only='True')
        checkpoint = ModelCheckpoint(
            filepath = filepath, 
            monitor='val_mask_dice_coef', 
            mode='max', 
            save_best_only='True')
        print('##################ttrain seg loss##################t')

    callback_lists = [checkpoint]

    model=sinnet(check_class,lr)

    if os.path.exists(filepath_exist):
        model.load_weights(filepath_exist)

    train_images,masks,classes,pt_list,pf_list,ht_list,hf_list,total_list = read_samples(image_dir,mask_dir,method)
    val_list=pt_list[int(start*len(pt_list)):int(stop*len(pt_list))]+pf_list[int(start*len(pf_list)):int(stop*len(pf_list))]+ht_list[int(start*len(ht_list)):int(stop*len(ht_list))]+hf_list[int(start*len(hf_list)):int(stop*len(hf_list))]
    train_list=list(set(total_list).difference(set(val_list)))

    prostatex_val_list=pt_list[int(start*len(pt_list)):int(stop*len(pt_list))]+pf_list[int(start*len(pf_list)):int(stop*len(pf_list))]
    hk_val_list=ht_list[int(start*len(ht_list)):int(stop*len(ht_list))]+hf_list[int(start*len(hf_list)):int(stop*len(hf_list))]

    model.fit(
        np.array([train_images[i] for i in train_list]).reshape(-1,224,224,1),
        {"class" : np.array([classes[i] for i in train_list]).reshape(-1,2), 
        "mask" : np.array([masks[i] for i in train_list]).reshape(-1,224,224,1)
        },
        epochs=epoch,
        batch_size=2,
        validation_data=(np.array([train_images[i] for i in val_list]).reshape(-1,224,224,1),
        {"class" : np.array([classes[i] for i in val_list]).reshape(-1,2), 
        "mask" : np.array([masks[i] for i in val_list]).reshape(-1,224,224,1)}
        ),
        shuffle=True,
        callbacks=callback_lists)

    ###############

    model = load_model(filepath, compile=False)
    ###############

    model.save_weights('{0}_A_joint_{2}/{1}_sinnet_{0}_weights.h5'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method))

    predictions_prostatex=model.predict(np.array([train_images[i] for i in prostatex_val_list]).reshape(-1,224,224,1))
    label_prostatex=np.array([classes[i] for i in prostatex_val_list]).reshape(-1,2)
    auc=roc_auc_score(label_prostatex[:,0],predictions_prostatex[1][:,0])
    print('prostatex_auc:',auc)
    conf_mat=confusion_matrix(np.argmax(np.array([classes[i] for i in prostatex_val_list]).reshape(-1,2),axis=1),np.argmax(predictions_prostatex[1],axis=1))
    
    print('sensitivity:', conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),'\n')
    print('specificity:', conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),'\n')
    print('acc:', (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0]),'\n')

    
    predictions_hk=model.predict(np.array([train_images[i] for i in hk_val_list]).reshape(-1,224,224,1))
    label_hk=np.array([classes[i] for i in hk_val_list]).reshape(-1,2)
    auc=roc_auc_score(label_hk[:,0],predictions_hk[1][:,0])
    print('hk_auc:',auc)
    conf_mat=confusion_matrix(np.argmax(np.array([classes[i] for i in hk_val_list]).reshape(-1,2),axis=1),np.argmax(predictions_hk[1],axis=1))
    
    print('sensitivity:', conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1]),'\n')
    print('specificity:', conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1]),'\n')
    print('acc:', (conf_mat[0][0]+conf_mat[1][1])/(conf_mat[0][0]+conf_mat[1][1]+conf_mat[0][1]+conf_mat[1][0]),'\n')


    np.savetxt('{0}_A_joint_{2}/px_{1}_label.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method), label_prostatex)
    np.savetxt('{0}_A_joint_{2}/px_{1}_predictions.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method), predictions_prostatex[1])

    np.savetxt('{0}_A_joint_{2}/hk_{1}_predictions.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method), predictions_hk[1])
    np.savetxt('{0}_A_joint_{2}/hk_{1}_label.txt'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method), label_hk)

    model.save_weights('{0}_A_joint_{2}/{1}_sinnet_{0}_weights.h5'.format(sequence, str(int(start * 10)) + '_' + str(int(stop * 10)),method))

    del model
    K.clear_session()
    gc.collect()
    return start,stop



if __name__ == '__main__':

    ###############
    methods = ['scale','standardization','scale_BFC','standardization_BFC','scale_NF','standardization_NF']

    start = 0.6
    stop = 0.8
    
    epoch = 50
    check_class = 1  
    method = 'scale'
    lr = 1e-6
    

    sequence = 'T2'
    print(sequence,start,stop)
    image_dir = '../data/X_A/T2/images'
    mask_dir = '../data/X_A/T2/labels'
    if 'BFC' in method:
        image_dir = image_dir.replace('images','images_BFC')
    train_and_predict(image_dir,mask_dir,start,stop,sequence,check_class,epoch,lr,method)
