import shutil
import os

"""
function: 
- clear_old_files: remove existing files 
- copy_new_files: copy source file to target address
"""


def clear_old_files(output_path):
    for file in os.listdir(output_path + 'T2/'):
        os.remove(os.path.join(output_path + 'T2/', file))
    for file in os.listdir(output_path + 'BVAL/'):
        os.remove(os.path.join(output_path + 'BVAL/', file))
    for file in os.listdir(output_path + 'ADC/'):
        os.remove(os.path.join(output_path + 'ADC/', file))


def copy_new_files(src_address, target_address):
    shutil.copy(src_address, target_address)


if __name__ == "__main__":
    clear_old_files('./media/upload/')
    copy_new_files('./image/T2IM-0001-0020.dcm', './media/upload/T2')
