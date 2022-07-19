# domain-adaptation-lesion-assessment

Testing Your Own Data using the Prostate Lesion Detection and Classification (PLDC) Software:

1. To realize automatic PLDC using multi-cohort mpMRIs (T2, ADC, and hDWI), please download the testing software (call_test.exe) first in the following link: https://www.dropbox.com/sh/v5r9ar40oo74ljw/AAA79IIF4WV3n0TbMJmwMu7Na?dl=0

2. unzip the folder ./PLDC_software.zip, put the downloaded call_test.exe under ./PLDC_software/

3. Intsall packages for training and software testing. The install requirements:
python=3.6.5
keras
tensorflow = 1.15
opencv-python
pydicom
numpy 
pillow
scikit-image
SimpleITK

4. Prepare your multi-cohort mpMRI dicoms (Dicoms are required as the meta data will be used for prostate cropping on ADC and hDWI). You can regard a dataset as the source domain (e.g., the public dataset PROSTATEx), and your mpMRI-based local cohort dataset as the target domain. You will test the target samples using the software with your well-trained domain adaptation (DA) model by the following steps.

5. Train the prostate segmentation model using T2 images with codes under ./maskrcnn_model. Save the best prostate segmentation model (i.e., weight1.h5).

6. With the pre-cropped prostate regions on T2 images using the prostate segmentation model, generate the weak labels for the mpMRIs. 

7. Train DA model for mpMRI-based PLDC using source and target samples. 
First, you are adviced to train the segmentor module using codes under ./joint_model with the combined samples from both domains, and save the best model of lesion segmentor. Then, train the lesion segmentor and maligancy classifier modules simultaneously (initialized with the best weight of the lesion segmentor), and save the best PLDC models (i.e., T2: weight2_1.h5, ADC: weight3_1.h5, hDWI:  weight4_1.h5). Lastly, initialized with the best PLDC models in each branch of the corresponding DA models, train each DA model with all the three modules simultaneously, including the domain transfer module, and then save best DA models (i.e., T2: weight2_2.h5, ADC: weight3_2.h5, hDWI: weight4_2.h5).

8. Put all the well-trained weights under the folder ./PLDC_software/doc/weights/

9. Begin to test your mpMRIs from target domain. Open the call_test.exe, and you can start your testing via: "Main menu" → "Start testing". You can find the predicted results under the folder "./media/output/", including prostate segmentation, prostate lesion detection, and lesion malignancy results.

Note: 1) Please do not delete any of the existing folders of this software, such as image, doc folders.
      2) For image input, only T2 is allowed, as ADC and hDWI need DICOM metadata for prostate region registration. Make sure your input image is a 3-channel image.
      3) For DICOM input, T2 is necessary, while ADC and hDWI are optional.
      4) We provide several samples from PROSTATEx (source domain) under ./test_samples, while the target samples are not open-sourced here due to the patient privacy and ethical issues. Our models trained with PROSTATEx (source domain) and local target dataset are also provided for your reference. The total size of this software is around 6GB, including the executable software (~699MB) and model weights (~3.8G). The testing weights trained by our local cohort data (./PLDC_software/doc/weights/) can be downloaded in the following link: https://www.dropbox.com/sh/v5r9ar40oo74ljw/AAA79IIF4WV3n0TbMJmwMu7Na?dl=0
