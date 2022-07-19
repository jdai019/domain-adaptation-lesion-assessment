import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from prostate import Ui_ProstateSoftware
from Myfunction import clear_old_files, copy_new_files
from test_DICOM import predict_dicom
from test_image import predict_img
import time


class ProstateApp(QMainWindow, Ui_ProstateSoftware):
    def __init__(self, parent=None):
        super(ProstateApp, self).__init__(parent)
        self.setupUi(self)
        self.address_dicom_T2 = 'address'
        self.address_dicom_ADC = 'address'
        self.address_dicom_BVAL = 'address'
        self.address_img = 'address'
        self.target_address_upload_T2 = './media/upload/T2'
        self.target_address_upload_BVAL = './media/upload/BVAL'
        self.target_address_upload_ADC = './media/upload/ADC'
        # connect the function to the component
        self.actionHomepage.triggered.connect(self.go_to_homepage)
        self.actionsDicom.triggered.connect(self.go_to_dicom_test)
        self.actionUpload_img_file.triggered.connect(self.go_to_img_test)
        # push button: choose file
        self.pushButton_chooseT2.clicked.connect(self.slot_btn_chooseFile_dicom_T2)
        # self.pushButton_chooseT2.clicked.connect(self.start_progress)
        self.pushButton_chooseADC.clicked.connect(self.slot_btn_chooseFile_dicom_ADC)
        self.pushButton_chooseBVAL.clicked.connect(self.slot_btn_chooseFile_dicom_BVAL)
        self.pushButton_choose_img.clicked.connect(self.slot_btn_chooseFile_img)
        # push button: submit
        self.pushButton_submit_dicom.clicked.connect(self.slot_btn_submit_dicom)
        self.pushButton_submit_img.clicked.connect(self.slot_btn_submit_img)

    def go_to_homepage(self):
        self.stackedWidget.setCurrentIndex(0)

    def go_to_dicom_test(self):
        self.stackedWidget.setCurrentIndex(1)
        clear_old_files('./media/upload/')
        self.label_uploadResult_dicomT2.setText("")
        self.label_uploadResult_dicomADC.setText("")
        self.label_uploadResult_dicomBVAL.setText("")
        self.label_waitFlag_dicom.setText("")

    def go_to_img_test(self):
        self.stackedWidget.setCurrentIndex(2)
        clear_old_files('./media/upload/')
        self.label_uploadResult_img.setText("")
        self.label_waitFlag_img.setText("")

    def go_to_result_dicom(self):
        self.stackedWidget.setCurrentIndex(3)

    def go_to_result_img(self):
        self.stackedWidget.setCurrentIndex(4)

    def slot_btn_chooseFile_dicom_T2(self):

        file_name_choose, file_type = QFileDialog.getOpenFileName(self,
                                                                  caption="Choose file",
                                                                  directory='.',
                                                                  filter='Dicom (*.dcm)')
        if file_name_choose == '':
            print("\n cancel choosing")
            return
        print("\n You have choosed:")
        print(file_name_choose)
        self.label_uploadResult_dicomT2.setText("T2 upload successfully")
        self.address_dicom_T2 = file_name_choose
        copy_new_files(self.address_dicom_T2, self.target_address_upload_T2)

    def slot_btn_chooseFile_dicom_BVAL(self):
        file_name_choose, file_type = QFileDialog.getOpenFileName(self,
                                                                  caption="Choose file",
                                                                  directory='.',
                                                                  filter='Dicom (*.dcm)')
        if file_name_choose == '':
            print("\n cancel choosing")
            return
        print("\n You have choosed:")
        print(file_name_choose)
        self.label_uploadResult_dicomBVAL.setText("BVAL upload successfully")
        self.address_dicom_BVAL = file_name_choose
        copy_new_files(self.address_dicom_BVAL, self.target_address_upload_BVAL)

    def slot_btn_chooseFile_dicom_ADC(self):
        file_name_choose, file_type = QFileDialog.getOpenFileName(self,
                                                                  caption="Choose file",
                                                                  directory='.',
                                                                  filter='Dicom (*.dcm)')
        if file_name_choose == '':
            print("\n cancel choosing")
            return
        print("\n You have choosed:")
        print(file_name_choose)
        self.label_uploadResult_dicomADC.setText("ADC upload successfully")
        self.address_dicom_ADC = file_name_choose
        copy_new_files(self.address_dicom_ADC, self.target_address_upload_ADC)


    def slot_btn_chooseFile_img(self):
        file_name_choose, file_type = QFileDialog.getOpenFileName(self,
                                                                  caption="Choose file",
                                                                  directory='.',
                                                                  filter='Images (*.jpg *.jpeg *.png)')

        if file_name_choose == '':
            print("\n cancel choosing")
            return
        print("\n You have choosed:")
        print(file_name_choose)
        self.label_uploadResult_img.setText("img upload successfully")
        self.address_img = file_name_choose
        copy_new_files(self.address_img, self.target_address_upload_T2)

    def slot_btn_submit_dicom(self):
        """
        1. go to result page
        2. call prediction model
        3. display prediction result: malignancy/benign
        4. display images: raw, contour, ROI
        """
        self.label_waitFlag_dicom.setText("Testing, please wait...")
        self.label_waitFlag_dicom.repaint()
        # step 1: call prediction model
        result = predict_dicom()
        # step 2: go to result page
        self.go_to_result_dicom()
        # step 3: display prediction result: malignant/benign
        self.label_predictResult_dicom.setText(result)
        # step 4: display images: raw, contour, ROI.
        # The three of T2 would be displayed compulsively
        # The raw, contour, ROI of ADC or BVAL would be displayed if they exist
        self.label_RawT2.setPixmap(QPixmap('./media/output/raw.png'))
        self.label_contourT2.setPixmap(QPixmap('./media/output/contour.png'))
        self.label_ROIT2.setPixmap(QPixmap('./media/output/prostate_roi.png'))

        self.label_RawADC.setPixmap(QPixmap('./media/output/adc.png'))
        self.label_ContourADC.setPixmap(QPixmap('./media/output/adc_contour.png'))
        self.label_ROIADC.setPixmap(QPixmap('./media/output/adc_crop.png'))

        self.label_RawBVAL.setPixmap(QPixmap('./media/output/bval.png'))
        self.label_contourBVAL.setPixmap(QPixmap('./media/output/bval_contour.png'))
        self.label_ROIBVAL.setPixmap(QPixmap('./media/output/bval_crop.png'))

    def slot_btn_submit_img(self):
        """
        1. go to result page
        2. call prediction model
        3. display prediction result: malignancy/benign
        4. display images: raw, contour, ROI
        """
        self.label_waitFlag_img.setText("Testing, please wait...")
        self.label_waitFlag_img.repaint()
        # step 1: go to result page
        self.go_to_result_img()
        # step 2: call prediction model
        result = predict_img()
        # step 3: display prediction result: malignant/benign
        self.label_predictResult_img.setText(result)
        # step 4: display images: raw, contour, ROI.
        self.label_RawT2_img.setPixmap(QPixmap('./media/output/raw.png'))
        self.label_contourT2_img.setPixmap(QPixmap('./media/output/contour.png'))
        self.label_ROIT2_img.setPixmap(QPixmap('./media/output/prostate_roi.png'))


    # def start_progress(self):
    #     # self.progressBar_dicom.setVisible(True)
    #     max_value = 100
    #     self.progressBar_dicom.setMaximum(max_value)
    #     for i in range(max_value):
    #         time.sleep(0.1)
    #         self.progressBar_dicom.setValue(i + 1)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = ProstateApp()
    myWin.show()
    sys.exit(app.exec())
