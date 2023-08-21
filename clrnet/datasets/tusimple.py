import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from clrnet.utils.tusimple_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    'trainval':
    ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}


@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips',
                                                     'seg_label')[:-3] + 'png'
                # print(mask_path)
                # seg_label/0531/1492637053320698897/20.png
                
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                         for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                
                #print(lanes)
                #[[(560, 260), (524, 270), (488, 280), (452, 290), (416, 300), (380, 310), (344, 320), (308, 330), (272, 340), (236, 350), (200, 360), (164, 370), (129, 380), (93, 390), (57, 400), (21, 410)], 
                # [(612, 280), (601, 290), (591, 300), (580, 310), (569, 320), (559, 330), (548, 340), (538, 350), (527, 360), (517, 370), (506, 380), (495, 390), (485, 400), (474, 410), (464, 420), (453, 430), (442, 440), (432, 450), (421, 460), (411, 470), (400, 480), (389, 490), (379, 500), (368, 510), (358, 520), (347, 530), (336, 540), (326, 550), (315, 560), (305, 570), (294, 580), (284, 590), (273, 600), (262, 610), (252, 620), (241, 630), (231, 640), (220, 650), (209, 660), (199, 670), (188, 680), (178, 690), (167, 700), (156, 710)], 
                # [(726, 260), (749, 270), (771, 280), (794, 290), (817, 300), (839, 310), (862, 320), (884, 330), (907, 340), (930, 350), (952, 360), (975, 370), (998, 380), (1020, 390), (1043, 400), (1066, 410), (1088, 420), (1111, 430), (1134, 440), (1156, 450), (1179, 460), (1201, 470), (1224, 480), (1247, 490), (1269, 500)]]

                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path':
                    osp.join(self.data_root, data['raw_file']),
                    'img_name':
                    data['raw_file'],
                    'mask_path':
                    osp.join(self.data_root, mask_path),
                    'lanes':
                    lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

###################밑에 코드 일단은 안쓰는듯######################

    def pred2lanes(self, pred):
        print("what????????????")
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):

        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):

        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):

        pred_filename = os.path.join(output_basedir,
                                     'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename,
                                                self.cfg.test_json_file)
        self.logger.info(result)
        return acc
