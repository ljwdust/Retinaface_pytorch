from pathlib import Path

def get_fddb_dict():
    """FDDB数据集路径到文件夹的映射"""
    list_root = './data/FDDB/imglist/'
    img_dict = {}
    for txtpath in Path(list_root).glob('*.txt'):
        with open(txtpath, 'r', encoding='utf-8') as f:
            flist = f.readlines()
            flist = [l.strip() for l in flist]
            for i in flist:
                img_dict[i] = str(int(txtpath.stem.split('-')[-1]))
    return img_dict
