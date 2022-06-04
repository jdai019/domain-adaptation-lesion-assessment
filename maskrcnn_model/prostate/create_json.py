import json
import os

import numpy as np
from PIL import Image

from skimage import measure
import pydicom



class Jsonformat():
    def __init__(self,filename,regions,size):
        self.filename=filename
        self.regions = regions
        self.size=size

class Regions():
    def __init__(self, i):
        self.i=i

class I():
    def __init__(self, region_attributes,shape_attributes):
        self.region_attributes=region_attributes
        self.shape_attributes=shape_attributes

class Shape_attributes():
    def __init__(self,name,all_point_x,all_point_y):
        self.name=name
        self.all_point_x=all_point_x
        self.all_point_y=all_point_y


dict={}

base_dir = 'D:/prostate/i2cvb'
save_path='D:/prostate/mask_rcnn/val'
prostate_num = 0

machines = os.listdir(base_dir)
for machine in machines:
    machine_path = os.path.join(base_dir, machine)
    patients = os.listdir(machine_path)
    for patient in patients:
        patient_path = os.path.join(machine_path, patient)
        sequences = os.listdir(patient_path)
        for sequence in sequences:
            if 'GT' in sequence:
                sequence_path = os.path.join(patient_path, sequence)

                # save_image_path = save_path + '/' + machine + '/' + patient.split(' ')[1]
                # if not os.path.exists(save_image_path):
                #     os.makedirs(save_image_path)

                structures = os.listdir(sequence_path)
                for structure in structures:
                    if 'prostate' in structure:
                        prostate_path = os.path.join(sequence_path, structure)
                        dicoms = os.listdir(prostate_path)
                        for dicom in dicoms:
                            dicom_file = os.path.join(prostate_path, dicom)
                            data = pydicom.read_file(dicom_file)
                            prostate_array = data.pixel_array
                            if np.sum(prostate_array) == 0:
                                continue
                            else:

                                if not (os.path.exists(patient_path + '/T2W/' + dicom) and os.path.exists(sequence_path + '/cg/' + dicom) and os.path.exists(sequence_path + '/pz/' + dicom)):
                                    continue

                                if prostate_num%10==0:

                                    print(prostate_num, dicom_file)


                                    raw_dicom_path = patient_path + '/T2W/' + dicom


                                    raw_dicom_file = pydicom.read_file(raw_dicom_path)
                                    raw_array = raw_dicom_file.pixel_array
                                    raw_array = 255 * ((raw_array - np.amin(raw_array)) / (
                                    (np.amax(raw_array) - np.amin(raw_array))))
                                    raw_image = Image.fromarray(raw_array).convert('RGB')

                                    image_name=(raw_dicom_path.split('/')[2].split('\\')[2].split(' ')[1]+'_'+raw_dicom_path.split('/')[4]).replace('dcm','png')

                                    raw_image.save(save_path+'/'+image_name)

                                    size = os.path.getsize(save_path+'/'+image_name)

                                    object_num=1
                                    newregions = {}

                                    pz_dicom_path = sequence_path + '/pz/' + dicom
                                    pz_dicom_file = pydicom.read_file(pz_dicom_path)
                                    raw_pz = Image.fromarray(pz_dicom_file.pixel_array).convert('L')

                                    pz_contour = measure.find_contours(np.array(raw_pz), 100)
                                    if len(pz_contour)!=0:

                                        for c in pz_contour:
                                            c = np.around(c).astype(np.int)

                                        all_point_x =c[:,0].tolist()
                                        all_point_y =c[:,1].tolist()

                                        newshape_attributes = Shape_attributes('polygon', all_point_x, all_point_y)

                                        newi = I('pz', newshape_attributes.__dict__)

                                        newregions.update({str(object_num): newi.__dict__})

                                        object_num+=1

                                    cg_dicom_path = sequence_path + '/cg/' + dicom
                                    cg_dicom_file = pydicom.read_file(cg_dicom_path)
                                    raw_cg = Image.fromarray(cg_dicom_file.pixel_array).convert('L')

                                    cg_contour = measure.find_contours(np.array(raw_cg), 100)
                                    if len(cg_contour) != 0:

                                        for c in cg_contour:
                                            c = np.around(c).astype(np.int)

                                        all_point_x = c[:,0].tolist()
                                        all_point_y = c[:,1].tolist()

                                        newshape_attributes = Shape_attributes('polygon', all_point_x, all_point_y)

                                        newi = I('cg', newshape_attributes.__dict__)

                                        newregions.update({str(object_num): newi.__dict__})

                                        object_num += 1


                                    prostate_dicom_path = sequence_path + '/prostate/' + dicom
                                    prostate_dicom_file = pydicom.read_file(prostate_dicom_path)
                                    raw_prostate = Image.fromarray(prostate_dicom_file.pixel_array).convert('L')

                                    prostate_contour = measure.find_contours(np.array(raw_prostate), 100)

                                    if len(prostate_contour) != 0:

                                        for c in prostate_contour:
                                            c = np.around(c).astype(np.int)

                                        all_point_x = c[:,0].tolist()
                                        all_point_y = c[:,1].tolist()

                                        newshape_attributes = Shape_attributes('polygon', all_point_x, all_point_y)

                                        newi = I('prostate', newshape_attributes.__dict__)

                                        newregions.update({str(object_num): newi.__dict__})

                                        newjson = Jsonformat(image_name, newregions, size)

                                    dict.update({image_name: newjson.__dict__})

                                prostate_num += 1

print(len(dict))
json_data=json.dumps(dict)
json_train_file=save_path+'/0_val.json'
# with open(json_val_file, "w+") as f:
with open(json_train_file, "w+") as f:
    # print(json_data)
    f.write(json_data)
    # json.dump(json_data, f)
    f.close()