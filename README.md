# domain-adaptation-lesion-assessment

The software and the source code can be downloaded in the following link:
https://www.dropbox.com/sh/v5r9ar40oo74ljw/AAA79IIF4WV3n0TbMJmwMu7Na?dl=0


Install requirements:
python=3.6.5
keras
tensorflow = 1.15
opencv-python
pydicom
numpy 
pillow
scikit-image
SimpleITK


1. In call_test.exe, you can start your testing via: "Main menu" → "Start testing".
2. Please do not delete any of the existing folders of this software, such as image, doc folders.
3. You can find the predicted results under the folder "./media/output/", including prostate segmentation, prostate lesion detection, and lesion malignancy results.
4. For image input, only T2 is allowed, as ADC and hDWI need DICOM metadata for prostate region registration. Make sure your input image is a 3-channel image.
5. For DICOM input, T2 is necessary, while ADC and hDWI are optional.
6. The size of this software is around 6GB, including exe (699MB), model weights(~3.8G), etc.
7. Regarding training, you can first train the segmentor module using samples from both domains, and save the first best model. Then, train the segmentor and classifier modules simultaneously (initialized with the best segmenter weight), and save the second best model. Lastly, initialized with the second best model in each branch of the domain adaptation (i.e. DA) model, train the DA model with three modules, including the domain transfer module, and save best model.
