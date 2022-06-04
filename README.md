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


1. You can start your testing via: "Main menu" → "Start testing".
2. Please do not delete any of the existing folders/files, e.g., image, doc folder.
3. You can find the predicted results under the folder "./media/output/", including prostate segmentation, prostate lesion detection, and lesion malignancy results.
4. For image input, only T2 is allowed, as ADC and hDWI need DICOM metadata for prostate region registration. Make sure your input image is a 3-channel image.
5. For DICOM input, T2 is necessary, while ADC and hDWI are optional.
