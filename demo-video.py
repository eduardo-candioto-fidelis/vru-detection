import _init_path
import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import time
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, data_root_path=None, label_root_path=None, logger=None, ext='.bin'):
        """
        Args:
            data_root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=data_root_path, logger=logger
        )
        self.data_root_path = data_root_path
        self.ext = ext
        data_file_list = glob.glob(str(data_root_path / f'*{self.ext}')) if self.data_root_path.is_dir() else [self.data_root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

        self.label_root_path = label_root_path
        label_file_list = glob.glob(str(label_root_path / f'*.txt')) if self.label_root_path.is_dir() else [self.label_root_path]

        label_file_list.sort()
        self.label_file_list = label_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError
        
        labels = self._load_gt(self.label_file_list[index])

        input_dict = {
            'points': points,
            'labels': labels,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def _load_gt(self, gt_path):
        gt_boxes = []
        with open(gt_path, 'r') as fp:
            for line in fp:
                elements = line.split()
                numeric_gt_boxes = [float(x) for x in elements[:-1]]
                label = elements[-1]
                gt_boxes.append(numeric_gt_boxes)
        gt_boxes = np.array(gt_boxes)
        return gt_boxes


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--gt_path', type=str, default='ground_truth_data',
                        help='specify the ground truth bounding boxes')

    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


pause = False


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_root_path=Path(args.data_path), label_root_path=Path(args.gt_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    def toggle_pause(vis):
        global pause
        pause = not pause
    vis.register_key_callback(ord("Q"), toggle_pause)

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            if idx < 260:
                continue
            while pause:
                time.sleep(0.03)
                vis.poll_events()
                vis.update_renderer()
            time.sleep(0.1)
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_multiple_scenes(
                vis,
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                gt_boxes=data_dict['labels'][0]
            )
            #view_control = vis.get_view_control()
            #view_control.translate(0, -100)
            #view_control.rotate(0, 0) 

            vis.update_renderer()

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    vis.destroy_window()

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
