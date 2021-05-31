import os
import os.path as osp

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test
from my.data_process.convert_fair1m_to_mmdet import convert_dota_to_mmdet


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):
    dst_trainval_path = osp.join(dst_path, 'trainval_split')
    dst_test_base_path = osp.join(dst_path, 'test_split')
    dst_test_path = osp.join(dst_path, 'test_split')
    # make dst path if not exist
    mkdir_if_not_exists(dst_path)
    mkdir_if_not_exists(dst_trainval_path)
    mkdir_if_not_exists(dst_test_base_path)
    mkdir_if_not_exists(dst_test_path)

    # split train data
    print('split train data')
    split_train = splitbase_trainval(osp.join(src_path, 'train'), dst_trainval_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_train.splitdata(scale)
    print('split val data')
    # split val data
    split_val = splitbase_trainval(osp.join(src_path, 'val'), dst_trainval_path,
                                   gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_val.splitdata(scale)
    # split test data
    print('split test data')
    # todo:原来为splitbase_test,'test/images'
    split_test = splitbase_trainval(osp.join(src_path, 'test'), dst_test_path,
                                gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_test.splitdata(scale)

    convert_dota_to_mmdet(dst_trainval_path,
                          osp.join(dst_trainval_path, 'trainval_s2anet.pkl'))
    convert_dota_to_mmdet(dst_test_base_path,
                          osp.join(dst_test_base_path, 'test_s2anet.pkl'), trainval=False)
    print('done!')


if __name__ == '__main__':
    prepare_multi_scale_data('/home/jyc/arashi/data/FAIR1M_dataset_small', '/home/jyc/arashi/data/FAIR1M_dataset_small_1024', gap=100, subsize=800,scales=[1.0],
                             num_process=32)
