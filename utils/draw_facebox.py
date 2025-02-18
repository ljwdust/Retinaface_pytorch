import argparse
import cv2
import numpy as np
import multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_color(color_info):
    color_dict = {
        'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)
    }
    if isinstance(color_info, str):
        return color_dict[color_info]
    elif isinstance(color_info, tuple):
        return color_info
    elif isinstance(color_info, list):
        tuple(color_info)
    else:
        raise TypeError('Invalid type of parameter color_info.')


def draw_facebox(img, dets, vis_thres, color=(0, 0, 255), draw_landmark=False):
    """在图像上画出人脸框
    img: 图像路径/PIL.Image格式/numpy格式
    dets: 人脸框数据，每组数据为(x1,y1,x2,y2,score)，(x1,y1)为左上角坐标，(x2,y2)为右下角坐标，score为置信度
    vis_thres: 可视化的人脸阈值，范围0~1
    color: 人脸框颜色，格式(Blue,Greeb,Red)，默认红色
    draw_landmark: 是否可视化5个特征点
    """
    if isinstance(img, str) or isinstance(img, Path):
        img = cv2.imread(str(img), cv2.IMREAD_COLOR)

    if isinstance(img, Image.Image):
        img = np.array(img)
        img = img[:,:,::-1].copy() # convert to BGR

    color = get_color(color)
    box_line_width = max(round(sum(img.shape) / 2 * 0.002), 2)  # line width
    dets = np.array(dets)
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        # draw box
        b = list(map(int, b.round()))
        p1, p2 = (b[0], b[1]), (b[2], b[3])
        cv2.rectangle(img, p1, p2, color, box_line_width)
        # draw score
        box_width = p2[0] - p1[0]
        fontScale = max(box_width * 0.008, 1) # font scale
        thickness = max(round(box_width * 0.02), 2) # font thickness
        w, h = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=thickness)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, text, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 
                    0, fontScale, (255, 255, 255),
                    thickness=thickness, lineType=cv2.LINE_AA)

        if draw_landmark:
            # landms
            cv2.circle(img, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img, (b[13], b[14]), 1, (255, 0, 0), 4)
    return img


def load_face_txt(txtpath):
    dets = []
    with open(str(txtpath), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines[2:]:
        det = line.strip().split()
        det = [float(d) for d in det]
        det[2] = det[0] + det[2]
        det[3] = det[1] + det[3]
        dets.append(det)
    return dets


def draw_by_txt(data_name, txt_root, vis_thres, save_root, color=(0, 0, 255), draw_landmark=False):
    """人脸框可视化，根据人脸检测txt结果
    data_name: 图像集名称，[widerface, FDDB, non_face]
    txt_root: 人脸检测txt结果路径
    vis_thres: 可视化的人脸阈值，范围0~1
    save_root: 可视化的图像保存路径
    color: 人脸框颜色，格式(Blue,Greeb,Red)，默认红色
    draw_landmark: 是否可视化5个特征点
    """
    # pool = mp.Pool(8)
    # pool.imap(draw_facebox, get())
    Path(save_root).mkdir(parents=True, exist_ok=True)
    if data_name == 'widerface':
        filelistpath = './data/widerface/val/filelist.txt'
        with open(filelistpath, 'r', encoding='utf-8') as f:
            img_paths = f.readlines()
            img_paths = [path.strip() for path in img_paths]
        for path in tqdm(img_paths, desc=f"Draw face box of {Path(txt_root).stem}"):
            imgpath = Path('./data/widerface/val/images').joinpath(path)
            txtpath = Path(txt_root).joinpath(path).with_suffix('.txt')
            dets = load_face_txt(txtpath)
            img_facebox = draw_facebox(imgpath, dets, vis_thres, color)
            save_path = Path(save_root).joinpath(txtpath.stem).with_suffix('.jpg').as_posix()
            cv2.imwrite(save_path, img_facebox)
    elif data_name == 'FDDB':
        from get_img_dict import get_fddb_dict
        img_dir_dict = get_fddb_dict()
        filelistpath = './data/FDDB/img_list.txt'
        with open(filelistpath, 'r', encoding='utf-8') as f:
            img_paths = f.readlines()
            img_paths = [path.strip() for path in img_paths]
        for path in tqdm(img_paths, desc=f"Draw face box of {Path(txt_root).stem}"):
            imgpath = Path('./data/FDDB/images').joinpath(path).with_suffix('.jpg')
            txtpath = Path(txt_root).joinpath(img_dir_dict[path]).joinpath(path.replace('/','_')).with_suffix('.txt')
            dets = load_face_txt(txtpath)
            img_facebox = draw_facebox(imgpath, dets, vis_thres, color)
            save_path = Path(save_root).joinpath(txtpath.stem).with_suffix('.jpg').as_posix()
            cv2.imwrite(save_path, img_facebox)
    elif data_name == 'non_face':
        filelistpath = './data/non_face/filelist.txt'
        with open(filelistpath, 'r', encoding='utf-8') as f:
            img_paths = f.readlines()
            img_paths = [path.strip() for path in img_paths]
        for path in tqdm(img_paths, desc=f"Draw face box of {Path(txt_root).stem}"):
            imgpath = Path('./data/non_face/images').joinpath(path)
            txtpath = Path(txt_root).joinpath(path).with_suffix('.txt')
            dets = load_face_txt(txtpath)
            img_facebox = draw_facebox(imgpath, dets, vis_thres, color)
            save_path = Path(save_root).joinpath(txtpath.stem).with_suffix('.jpg').as_posix()
            cv2.imwrite(save_path, img_facebox)
    else:
        raise RuntimeError('Invalid variable: data_name.')


def join_pictures(data_name, img_name):
    """多个人脸检测结果合并为一张图
    data_name: 数据集名称，[FDDB, wider_val, non_face]
    img_name: 图像名称
    """
    image_root = Path(__file__).absolute().parent.parent.joinpath('results/images')
    # 不同结果前缀，以及在图像上的水印文字
    prefix_list = {
        'founder': 'Founder v1', 
        'mnet0.25_nonface': 'Founder v2', 
        'baidu': 'Baidu', 
        'ali': 'Ali', 
        'tencent' : 'Tencent', 
        'face++': 'Face++'
    }
    # 图像左上角画prefix水印
    img_list = []
    for prefix, name in prefix_list.items():
        path = image_root.joinpath(prefix + '_' + data_name + '_img').joinpath(img_name).with_suffix('.jpg')
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        im_height, im_width, _ = img.shape
        fontScale = max(im_width * 0.0008, 1) # font scale
        thickness = max(round(im_width * 0.002), 2) # font thickness
        w, h = cv2.getTextSize(name, 0, fontScale=fontScale, thickness=thickness)[0]  # text width, height
        cv2.rectangle(img, (0,0), (w,h), (255,255,255), -1, cv2.LINE_AA)  # filled
        cv2.putText(img, name, (0, h), 
                    0, fontScale, (0, 0, 255),
                    thickness=thickness, lineType=cv2.LINE_AA)
        img_list.append(img)

    # 合并图像
    if len(img_list) % 2 == 1:
        blackimg = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        img_list.append(blackimg)
    line_cnt = int(len(img_list) / 2)
    img_line_1 = img_list[0].copy()
    for i in range(line_cnt-1):
        img_line_1 = np.hstack((img_line_1, img_list[i+1]))
    img_line_2 = img_list[line_cnt].copy()
    for i in range(line_cnt-1):
        img_line_2 = np.hstack((img_line_2, img_list[i+1+line_cnt]))
    joinedimg = np.vstack((img_line_1, img_line_2))
    save_path = image_root.joinpath('joined_images/' + data_name + '_' + img_name + '.jpg').as_posix()
    cv2.imwrite(save_path, joinedimg)
    print(f"Joined image saved to {save_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_name', default='widerface', type=str, help='[widerface, FDDB, non_face]')
    # parser.add_argument('--txt_root', default='', type=str, help='.txt results root')
    # parser.add_argument('--save_root', default='', type=str, help='drawing images saved root')
    # parser.add_argument('--vis_thres', default=0.1, type=float, help='visualization threshold')
    # args = parser.parse_args()
    # draw_by_txt(args.data_name, args.txt_root, args.vis_thres, args.save_root)
    
    join_pictures('wider_val', '3_Riot_Riot_3_522')
