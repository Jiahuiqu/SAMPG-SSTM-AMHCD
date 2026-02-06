import os
from scipy.io import loadmat



def get_Hermiston_dataset(bx):
    """
    307 × 241 × 154, 57311 unchanged, 16676 changed
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    if bx == '0':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/T1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/T2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/REF1.mat')

    # 随机变形
    elif bx == '1':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/1_USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/1_USAT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/1_USAREF.mat')
    elif bx == '3':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/3_USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/3_USAT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/3_USAREF.mat')
    elif bx == '5':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/5_USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/5_USAT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/5_USAREF.mat')
    # #旋转角度
    elif bx =='sj':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAREF.mat')

    elif bx =='reg':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/mobilesam/results/USA/warped_hyperspectral.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAREF.mat')

    elif bx =='0_reg':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/T1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/T2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/USA/REF1.mat')

    elif bx =='sj_noreg' or bx == 'affine':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/mobilesam/results/USA/warped_hyperspectral.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAREF.mat')

    elif bx =='sj_nopz':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/USA/USAREF.mat')

    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    img_t1 = data_t1['HypeRvieW'].astype('float32')

    if bx == '0' or bx == '0_reg':
        img_gt = data_gt['Binary'].astype('float32')
        img_t2 = data_t2['HypeRvieW'].astype('float32')
    elif bx == 'reg' or bx == 'sj_noreg' or bx == 'affine':
        img_t2 = data_t2['warped_hs'].astype('float32')
        img_gt = data_gt['HypeRvieW'].astype('float32')
    else:
        img_gt = data_gt['HypeRvieW'].astype('float32')
        img_t2 = data_t2['HypeRvieW'].astype('float32')
    return img_t1, img_t2, img_gt



def get_BayArea_dataset(bx):
    """
    600 × 500 × 224, 34211 unchanged, 39270 changed, 226519 undetermined
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    if bx == '0':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAY/Q1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAY/Q2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAY/REF.mat')

    # 旋转角度
    elif bx == '1':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/1_BAYT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/1_BAYT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/1_BAYREF.mat')
    elif bx == '3':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/3_BAYT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/3_BAYT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/3_BAYREF.mat')
    elif bx == '5':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/5_BAYT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/5_BAYT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/3_BAYREF.mat')

    # 随机变形
    elif bx == 'sj':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/BAYT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/BAYT2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/BAYREF.mat')

    elif bx == 'reg' or bx == 'affine':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/BAYT1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/mobilesam/results/BAY/warped_hyperspectral.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAY/BAYREF.mat')

    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    if bx == 'reg' or bx == 'affine':
        img_t1 = data_t1['HypeRvieW'].astype('float32')
        img_t2 = data_t2['warped_hs'].astype('float32')
        img_gt = data_gt['HypeRvieW'].astype('float32')
    else:
        img_t1 = data_t1['HypeRvieW'].astype('float32')
        img_t2 = data_t2['HypeRvieW'].astype('float32')
        img_gt = data_gt['HypeRvieW'].astype('float32')

    return img_t1, img_t2, img_gt


def get_Barbara_dataset(bx):
    """
    984 × 740 × 224, 80418 unchanged, 52134 changed, 595608 undetermined
    :return: img_t1, img_t2, img_gt
    """
    current_path = os.getcwd()
    # 无形变
    if bx == '0':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/Q1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/Q2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/REF.mat')

    # 旋转角度
    elif bx == '1':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/1_BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/1_BART2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/1_BARREF.mat')
    elif bx == '3':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/3_BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/3_BART2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/3_BARREF.mat')
    elif bx == '5':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/5_BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/5_BART2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/5_BARREF.mat')

    # 随机变形
    elif bx == 'sj':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BARREF.mat')

    elif bx == 'reg' or bx == 'affine':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/mobilesam/results/BAR/warped_hyperspectral.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BARREF.mat')

    elif bx == '0_reg':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/Q1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/Q2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/data/BAR/REF.mat')

    elif bx == 'sj_noreg':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/mobilesam/results/BAR/warped_hyperspectral.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BARREF.mat')

    elif bx == 'sj_nopz':
        data1_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART1.mat')
        data2_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BART2.mat')
        data3_path = os.path.join(current_path, '/media/xd132/E/ZTZ/SAMReg/datas/BAR/BARREF.mat')


    data_t1 = loadmat(data1_path)
    data_t2 = loadmat(data2_path)
    data_gt = loadmat(data3_path)

    if bx == 'reg' or bx == 'sj_noreg' or bx == 'affine':
        img_t1 = data_t1['HypeRvieW'].astype('float32')
        img_t2 = data_t2['warped_hs'].astype('float32')
        img_gt = data_gt['HypeRvieW'].astype('float32')
    else:
        img_t1 = data_t1['HypeRvieW'].astype('float32')
        img_t2 = data_t2['HypeRvieW'].astype('float32')
        img_gt = data_gt['HypeRvieW'].astype('float32')
    return img_t1, img_t2, img_gt


def get_dataset(dataset_name, bx):

    if dataset_name == 'Hermiston':
        return get_Hermiston_dataset(bx)
    elif dataset_name == 'BayArea':
        return get_BayArea_dataset(bx)
    elif dataset_name == 'Barbara':
        return get_Barbara_dataset(bx)


if __name__ == '__main__':
    img_t1, img_t2, img_gt = get_dataset('a')
