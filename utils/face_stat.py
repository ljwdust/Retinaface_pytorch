# 统计监测出的人脸数量
from pathlib import Path
from tqdm import tqdm

def counter(txt_root, score_thres):
    txtlist = list(Path(txt_root).glob('**/*.txt'))
    face_count = {}
    for txt in txtlist:
        cls = txt.parent.name.split('.')[1]
        with open(txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        faceNum = int(lines[1].strip())
        if faceNum > 0:
            faces = lines[2:]
            cnt = 0
            for face in faces:
                score = float(face.split()[4])
                if score >= score_thres:
                    cnt += 1
            if cls not in face_count:
                face_count[cls] = cnt
            else:
                face_count[cls] += cnt
    return face_count



# txt_root = 'ali_non_face_txt'
# cnt_list = []
# for i in tqdm(range(100)):
#     score_thres = (i+1) / 100
#     face_count = counter(txt_root, score_thres)
#     allcnt = sum(list(face_count.values()))
#     cnt_list.append(allcnt)

# print(cnt_list)


score_thres = 0.9

txt_root = '../results/mnet0.25_rotate_nonface-non_face_txt'
face_count = counter(txt_root, score_thres)
allcnt = sum(list(face_count.values()))
print(f"All count: {allcnt}")
print(face_count)

# # txt_root = 'retina_non_face_txt'
# txt_root = 'retina_detect_non_face_txt'
# face_count = counter(txt_root, score_thres)
# allcnt = sum(list(face_count.values()))
# print(f"All count: {allcnt}")
# print(face_count)

# txt_root = 'retina_new_detect_non_face_txt'
# face_count = counter(txt_root, score_thres)
# allcnt = sum(list(face_count.values()))
# print(f"All count: {allcnt}")
# print(face_count)
