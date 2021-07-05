import os
import os.path as osp

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test
from my.data_process.convert_fair1m_to_mmdet import convert_dota_to_mmdet


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path,word_name, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):
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
                                     gap=gap, subsize=subsize, num_process=num_process,ext=".bmp")
    for scale in scales:
        split_train.splitdata(scale)
    print('split val data')
    # split val data
    split_val = splitbase_trainval(osp.join(src_path, 'val'), dst_trainval_path,
                                   gap=gap, subsize=subsize, num_process=num_process,ext=".bmp")
    for scale in scales:
        split_val.splitdata(scale)
    # split test data
    print('split test data')
    # todo:原来为splitbase_test,'test/images'
    split_test = splitbase_trainval(osp.join(src_path, 'test'), dst_test_path,
                                gap=gap, subsize=subsize, num_process=num_process,ext=".bmp")
    for scale in scales:
        split_test.splitdata(scale)

    convert_dota_to_mmdet(dst_trainval_path,
                          osp.join(dst_trainval_path, 'trainval_s2anet.pkl'),word_name,ext=".bmp")
    convert_dota_to_mmdet(dst_test_base_path,
                          osp.join(dst_test_base_path, 'test_s2anet.pkl'), word_name, trainval=False,ext=".bmp")
    print('done!')


if __name__ == '__main__':
    word_name =['阿利伯克级驱逐舰', '圣安东尼奥级两栖船坞运输舰', '塔拉瓦级通用两栖攻击舰',
                      '琵琶形军舰', '企业级航母', '提康德罗加级巡洋舰', '佩里级护卫舰', '尾部OX头部圆指挥舰', '奥斯汀级两栖船坞运输舰', '潜艇', '惠德贝岛级船坞登陆舰',
                      '尼米兹级航母', '军舰', '俄罗斯库兹涅佐夫号航母', '医疗船', '中途号航母', '蓝岭级指挥舰', '小鹰级航母',
                      '航母']
    # word_name = ['warship', 'ship', 'aircraft_carrier', 'submarine']
    prepare_multi_scale_data('/home/jyc/arashi/data/HRSC2016_dataset', '/home/jyc/arashi/data/HRSC2016_dataset_800',
                             word_name,gap=100, subsize=800,scales=[1],
                             num_process=128)
