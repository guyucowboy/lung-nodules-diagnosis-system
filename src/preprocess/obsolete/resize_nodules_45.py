# encoding: utf-8
"""
@version: python 2.7
@software: PyCharm
"""

import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
import array
import math
import random
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x
# import traceback
workspace = '/home/mml/'
class nodules_crop(object):
    def __init__(self, workspace):
        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace,"incoming/train_subset14/")
        self.nodules_npy_path = os.path.join(self.workspace,"preprocess/3d_45/")
        self.all_annotations_mhd_path = os.path.join(self.workspace,"preprocess/mhd/")
        self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
        self.df_annotations = pd.read_csv(self.workspace + "tmp/csv/train/annotations.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
    #---各种预定义
    def set_window_width(self, image, MIN_BOUND=-1000.0):
        image[image < MIN_BOUND] = MIN_BOUND
        return image
    #---设置窗宽
    def resample(self,image, old_spacing, new_spacing=[1, 1, 1]):
        resize_factor = old_spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = old_spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        return image, new_spacing
    #---重采样
    def write_meta_header(self,filename, meta_dict):
        header = ''
        # do not use tags = meta_dict.keys() because the order of tags matters
        tags = ['ObjectType', 'NDims', 'BinaryData',
                'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
                'TransformMatrix', 'Offset', 'CenterOfRotation',
                'AnatomicalOrientation',
                'ElementSpacing',
                'DimSize',
                'ElementType',
                'ElementDataFile',
                'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
        for tag in tags:
            if tag in meta_dict.keys():
                header += '%s = %s\n' % (tag, meta_dict[tag])
        f = open(filename, 'w')
        f.write(header)
        f.close()
    def dump_raw_data(self,filename, data):
        """ Write the data into a raw format file. Big endian is always used. """
        #---将数据写入文件
        # Begin 3D fix
        data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
        # End 3D fix
        rawfile = open(filename, 'wb')
        a = array.array('f')
        for o in data:
            a.fromlist(list(o))
        # if is_little_endian():
        #    a.byteswap()
        a.tofile(rawfile)
        rawfile.close()
    def write_mhd_file(self,mhdfile, data, dsize):
        assert (mhdfile[-4:] == '.mhd')
        meta_dict = {}
        meta_dict['ObjectType'] = 'Image'
        meta_dict['BinaryData'] = 'True'
        meta_dict['BinaryDataByteOrderMSB'] = 'False'
        meta_dict['ElementType'] = 'MET_FLOAT'
        meta_dict['NDims'] = str(len(dsize))
        meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
        meta_dict['ElementDataFile'] = os.path.split(mhdfile)[1].replace('.mhd', '.raw')
        self.write_meta_header(mhdfile, meta_dict)
        pwd = os.path.split(mhdfile)[0]
        if pwd:
            data_file = pwd + '/' + meta_dict['ElementDataFile']
        else:
            data_file = meta_dict['ElementDataFile']
        self.dump_raw_data(data_file, data)
    def save_annotations_nodule(self,nodule_crop, name_index, cancer_flag):
        np.save(os.path.join(self.nodules_npy_path,"%01d_%06d_annotations.npy" % (cancer_flag,name_index)), nodule_crop)
        self.write_mhd_file(self.all_annotations_mhd_path + str(cancer_flag) + "_" + str(name_index) + '_annotations' + '.mhd', nodule_crop,nodule_crop.shape)
    #---保存结节文件，若需要使用Fiji软件查看分割效果可取消注释write_mhd_file
    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)
    #---匹配文件名
    def annotations_crop(self):
        for patient in enumerate(tqdm(self.ls_all_patients)):
            patient = patient[1]
            print(patient)
            # 检查这个病人有没有大于3mm的结节
            if patient not in self.df_annotations.file.values:
                print('Patient ' + patient + 'Not exist!')
                continue
            patient_nodules = self.df_annotations[self.df_annotations.file == patient]
            full_image_info = sitk.ReadImage(patient)
            full_scan = sitk.GetArrayFromImage(full_image_info)
            origin = np.array(full_image_info.GetOrigin())[::-1]  #---获取“体素空间”中结节中心的坐标
            old_spacing = np.array(full_image_info.GetSpacing())[::-1]  #---该CT在“世界空间”中各个方向上相邻单位的体素的间距
            image, new_spacing = self.resample(full_scan, old_spacing)#---重采样
            print('Resample Done')
            for index, nodule in patient_nodules.iterrows():
                nodule_center = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])#---获取“世界空间”中结节中心的坐标
                pos_v_center = np.rint((nodule_center - origin) / new_spacing)#映射到“体素空间”中的坐标
                pos_v_center = np.array(pos_v_center, dtype=int)
                #---这一系列的if语句是根据“判断一个结节的癌性与否需要结合该结节周边位置的阴影和位置信息”而来，故每个结节都获取了比该结节尺寸略大的3D体素
                if nodule.diameter_mm < 5:
                    window_size = 7
                elif nodule.diameter_mm < 10:
                    window_size = 9
                elif nodule.diameter_mm < 20:
                    window_size = 15
                elif nodule.diameter_mm < 25:
                    window_size = 17
                elif nodule.diameter_mm < 30:
                    window_size = 20
                else:
                    window_size = 22
                pos_zyx_1 = pos_v_center - window_size  # 注意是: Z, Y, X
                pos_zyx_2 = pos_v_center + window_size + 1
                pos_nodule_box = np.zeros([45,45,45],np.int16)#---nodule_box_size = 45
                pos_img_crop = image[pos_zyx_1[0]:pos_zyx_2[0], pos_zyx_1[1]:pos_zyx_2[1], pos_zyx_1[2]:pos_zyx_2[2]]#---截取立方体
                pos_img_crop = self.set_window_width(pos_img_crop)#---设置窗宽，小于-1000的体素值设置为-1000
                zeros_fill = int(math.floor((45 - (2*window_size+1))/2))
                try:
                    pos_nodule_box[zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill,zeros_fill:45 - zeros_fill] = pos_img_crop  # ---将截取的立方体置于nodule_box
                except:
                    # f = open("log.txt", 'a')
                    # traceback.print_exc(file=f)
                    # f.flush()
                    # f.close()
                    continue
                pos_nodule_box[pos_nodule_box == 0] = -1000#---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
                self.save_annotations_nodule(pos_nodule_box, index, 1)
                #---生成正样本
                neg_v_center=[random.randint(23,348),random.randint(23,327),random.randint(23,327)]
                neg_v_center=np.array(neg_v_center)
                window_size=22
                neg_zyx_1 = neg_v_center - window_size  # 注意是: Z, Y, X
                neg_zyx_2 = neg_v_center + window_size + 1
                neg_nodule_box = np.zeros([45, 45, 45], np.int16)  # ---nodule_box_size = 45
                neg_img_crop = image[neg_zyx_1[0]:neg_zyx_2[0], neg_zyx_1[1]:neg_zyx_2[1],neg_zyx_1[2]:neg_zyx_2[2]]  # ---截取立方体
                neg_img_crop = self.set_window_width(neg_img_crop)  # ---设置窗宽，小于-1000的体素值设置为-1000
                zeros_fill = int(math.floor((45 - (2 * window_size + 1)) / 2))
                try:
                    neg_nodule_box[zeros_fill:45 - zeros_fill, zeros_fill:45 - zeros_fill,zeros_fill:45 - zeros_fill] = neg_img_crop  # ---将截取的立方体置于nodule_box
                except:
                    # f = open("log.txt", 'a')
                    # traceback.print_exc(file=f)
                    # f.flush()
                    # f.close()
                    continue
                neg_nodule_box[neg_nodule_box == 0] = -1000  # ---将填充的0设置为-1000，可能有极少数的体素由0=>-1000，不过可以忽略不计
                self.save_annotations_nodule(neg_nodule_box, index, 0)
                #---生成负样本
            print('Done for this patient!\n\n')
        print('Done for all!')
    #---截取候选结节函数
if __name__ == '__main__':
    nc = nodules_crop(workspace)
    nc.annotations_crop()
