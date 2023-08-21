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



@DATASETS.register_module
class TuSimple(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.json_path = "/media/jimin/SSD8T/jimin/lane_data/Training/aihub_1900_1200"
        self.add_paths = ["c_1920_1200_daylight_train_1","c_1920_1200_daylight_train_2"]
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading aihub annotations...')
        self.data_infos = []
        max_lanes = 0

        for add_path in self.add_paths:
            label_dir = os.path.join(label_dir, "label", add_path)
            for filename in os.listdir(label_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(label_dir ,filename), 'r') as f:
                        json_data = json.load(f)

                        lanes = []
                        _lanes = []
                        for annotation in json_data['annotations']:
                            if annotation['class'] == 'traffic_lane':
                                lanes = [(point['x'], point['y']) for point in annotation['data']]
                                #lanes = [lane for lane in lanes if len(lane) > 0]
                            if (len(lanes) > 1):
                                _lanes.append(lanes)

                        file_name = json_data['image']['file_name']
                        seg_path = os.path.join(self.json_path, "seg", add_path)
                        img_path = os.path.join(self.json_path, "img", add_path)

                        max_lanes = max(max_lanes, len(_lanes))
                        self.data_infos.append({
                            'img_path':
                            osp.join(img_path, file_name),
                            'img_name':
                            file_name,
                            'mask_path':
                            osp.join(seg_path, file_name),
                            'lanes':
                            _lanes,
                        })
                        print(self.data_infos)
                        break

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes
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
