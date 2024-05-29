import cv2
import os
import glob

# 获取文件夹中所有的.jpg文件
need_writes=["depths","normals","renders"]
p1="output/100b2743-d/train/ours_30000/"
p2='output/1a543d27-5/train/ours_30000/'
p3="output/088db1d7-e/train/ours_30000/"
for need in need_writes:
    img_array = []
    path=f'{p3}{need}'
    image_pathes=os.listdir(path)
    image_pathes.sort()
    for filename in image_pathes:
        real_name=os.path.join(path,filename)
        img = cv2.imread(real_name)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

        # 创建视频写入对象
        out = cv2.VideoWriter(f'project{need}.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 10, size)

    # 将图片写入视频
    for i in range(len(img_array)):
        out.write(img_array[i])

    # 释放视频写入对象
    out.release()