import json
import os
import numpy as np
import cv2

json_path = "/media/jimin/SSD8T/jimin/lane_data/Training/aihub_1920_1200"

add_paths = ["c_1920_1200_daylight_train_1","c_1920_1200_daylight_train_2"]

lanes_coordinates = []
H, W = 1200, 1920
SEG_WIDTH = 40
cnt = 0
for add_path in add_paths:
    # seg_path = os.path.join(json_path, "seg", add_path)
    # /media/jimin/SSD8T/jimin/차선-횡단보도 인지 영상(수도권)/Training/aihub_1900_1200/seg/c_1920_1200_daylight_train_1
    # print(seg_path)
    label_dir = os.path.join(json_path, "label", add_path)
    for filename in os.listdir(label_dir):
        if filename.endswith(".json"):
            with open(os.path.join(label_dir ,filename), 'r') as f:
                json_data = json.load(f)
                lanes_coordinate = []
                lanes = []
                _lanes = []
                slope = [] 
                for annotation in json_data['annotations']:
                    if annotation['class'] == 'traffic_lane':
                        l = [(point['x'], point['y']) for point in annotation['data']]
                    if (len(l) > 1):
                        _lanes.append(l)
                        slope.append(
                            np.arctan2(l[-1][1] - l[0][1], l[0][0] - l[-1][0]) /
                            np.pi * 180)
                _lanes = [_lanes[i] for i in np.argsort(slope)]
                slope = [slope[i] for i in np.argsort(slope)]
                
                idx = [None for i in range(6)]
                for i in range(len(slope)):
                    if slope[i] <= 90:
                        idx[2] = i
                        idx[1] = i - 1 if i > 0 else None
                        idx[0] = i - 2 if i > 1 else None
                    else:
                        idx[3] = i
                        idx[4] = i + 1 if i + 1 < len(slope) else None
                        idx[5] = i + 2 if i + 2 < len(slope) else None
                        break
                for i in range(6):
                    lanes.append([] if idx[i] is None else _lanes[idx[i]])
                # print(lanes)

                img_path = json_data['image']['file_name']
                #print(img_path)
                seg_img = np.zeros((H, W, 3))
                list_str = []  # str to be written to list.txt
                for i in range(len(lanes)):
                    coords = lanes[i]
                    # if len(coords) < 4:
                    #     list_str.append('0')
                    #     continue
                    for j in range(len(coords) - 1):
                    #    print(coords[j], coords[j + 1])
                        #mini_h = abs(coords[j][0] - coords[j + 1][0])//10
                        cv2.line(seg_img, coords[j], coords[j + 1],
                                (i + 1, i + 1, i + 1), SEG_WIDTH // 2)
                    list_str.append('1')

                #seg_path = img_path.split("/")
                seg_path = os.path.join(json_path, "seg", add_path)
                os.makedirs(seg_path, exist_ok=True)
                file_name = json_data['image']['file_name']
                seg_path = os.path.join(seg_path, file_name[:-3] + "png")
                cv2.imwrite(seg_path, seg_img)
                cnt += 1
                print(cnt)


                #cv2.imwrite(seg_path, seg_img)

                # seg_path = "/".join([
                #     args.savedir, *img_path.split("/")[1:3], img_name[:-3] + "png"
                # ])
                # if seg_path[0] != '/':
                #     seg_path = '/' + seg_path
                # if img_path[0] != '/':
                #     img_path = '/' + img_path
                # list_str.insert(0, seg_path)
                # list_str.insert(0, img_path)
                # list_str = " ".join(list_str) + "\n"
                # list_f.write(list_str)

