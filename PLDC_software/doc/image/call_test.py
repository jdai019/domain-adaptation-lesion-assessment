import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
import PyQt5.QtCore
from PyQt5.QtCore import QCoreApplication
from prostate import Ui_ProstateSoftware
from Myfunction import clear_old_files, copy_new_files


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

    def go_to_img_test(self):
        self.stackedWidget.setCurrentIndex(2)
        clear_old_files('./media/upload/')

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
        self.address_img = file_name_choose
        copy_new_files(self.address_img, self.target_address_upload_T2)

    def slot_btn_submit_dicom(self):
        """
        1. go to result page
        2. call prediction model
        3. display prediction result: malignancy/benign
        4. display images: raw, contour, ROI
        """
        # step 1: go to result page
        self.go_to_result_dicom()
        # step 2: call prediction model
        # step 3: display prediction result: malignant/benign
        self.label_predictResult_dicom.setText('Malignant')
        # step 4: display images: raw, contour, ROI.
        # The three of T2 would be displayed compulsively
        # The raw, contour, ROI of ADC or BVAL would be displayed if they exist
        self.label_RawT2.setPixmap(QPixmap('./image/T2.jpg'))
        self.label_contourT2.setPixmap(QPixmap())
        self.label_ROIT2.setPixmap(QPixmap())

    def slot_btn_submit_img(self):
        """
        1. go to result page
        2. call prediction model
        3. display prediction result: malignancy/benign
        4. display images: raw, contour, ROI
        """
        # step 1: go to result page
        self.go_to_result_img()
        # step 2: call prediction model
        # step 3: display prediction result: malignant/benign
        self.label_predictResult_img.setText('Malignant')
        # step 4: display images: raw, contour, ROI.
        self.label_RawT2_img.setPixmap(QPixmap('./image/T2.jpg'))
        self.label_contourT2_img.setPixmap(QPixmap())
        self.label_ROIT2_img.setPixmap(QPixmap())


if __name__ == "__main__":
    QCoreApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    myWin = ProstateApp()
    myWin.show()
    sys.exit(app.exec())
