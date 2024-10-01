import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.io
import mmcv
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample
from mmengine.fileio import list_from_file
from mmengine.dataset import Compose, BaseDataset
from mmengine.runner import Runner
import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset
from mmaction.registry import DATASETS
from custom_dataset import DatasetZelda

# Import the custom dataset module
from custom_dataset import DatasetZelda  # Ensure this is the correct path to the custom_dataset.py file

# Pipeline definitions
@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    def transform(self, results):
        container = decord.VideoReader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        return results

@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results['total_frames']
        interval = total_frames // self.clip_len
        if self.test_mode:
            np.random.seed(42)
        inds_of_all_clips = []
        for i in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)
        results['frame_inds'] = np.concatenate(inds_of_all_clips)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results

@TRANSFORMS.register_module()
class VideoDecode(BaseTransform):
    def transform(self, results):
        frame_inds = results['frame_inds']
        container = results['video_reader']
        imgs = container.get_batch(frame_inds).asnumpy()
        imgs = list(imgs)
        results['video_reader'] = None
        del container
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results

@TRANSFORMS.register_module()
class VideoResize(BaseTransform):
    def __init__(self, r_size):
        self.r_size = (np.inf, r_size)

    def transform(self, results):
        img_h, img_w = results['img_shape']
        new_w, new_h = mmcv.rescale_size((img_w, img_h), self.r_size)
        imgs = [mmcv.imresize(img, (new_w, new_h)) for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results

@TRANSFORMS.register_module()
class VideoCrop(BaseTransform):
    def __init__(self, c_size):
        self.c_size = c_size

    def transform(self, results):
        img_h, img_w = results['img_shape']
        center_x, center_y = img_w // 2, img_h // 2
        x1, x2 = center_x - self.c_size // 2, center_x + self.c_size // 2
        y1, y2 = center_y - self.c_size // 2, center_y + self.c_size // 2
        imgs = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results

@TRANSFORMS.register_module()
class VideoFormat(BaseTransform):
    def transform(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        imgs = results['imgs']
        imgs = np.array(imgs)
        imgs = imgs.reshape((num_clips, clip_len) + imgs.shape[1:])
        imgs = imgs.transpose(0, 4, 1, 2, 3)
        results['imgs'] = imgs
        return results

@TRANSFORMS.register_module()
class VideoPack(BaseTransform):
    def __init__(self, meta_keys=('img_shape', 'num_clips', 'clip_len')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        inputs = to_tensor(results['imgs'])
        data_sample = ActionDataSample()
        data_sample.set_gt_label(results['label'])
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results

pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

pipeline = Compose(pipeline_cfg)
data_prefix = '/home/mler24_team001/.condor/DataSet_temp/train'
results = dict(filename=osp.join(data_prefix, 'Mitbrin_1_A15_B16_ID16.avi'), label=0)
packed_results = pipeline(results)

inputs = packed_results['inputs']
data_sample = packed_results['data_samples']

print('Shape of the inputs: ', inputs.shape)
print('Image shape: ', data_sample.img_shape)
print('Number of clips: ', data_sample.num_clips)
print('Clip length: ', data_sample.clip_len)
print('Label: ', data_sample.gt_label)

def clean_annotation_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8-sig') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            clean_line = line.strip()
            fout.write(clean_line + '\n')

input_ann_file = '/home/mler24_team001/.condor/DataSet_temp/train.txt'
output_ann_file = '/home/mler24_team001/.condor/DataSet_temp/train_clean.txt'
clean_annotation_file(input_ann_file, output_ann_file)

input_ann_file = '/home/mler24_team001/.condor/DataSet_temp/val.txt'
output_ann_file = '/home/mler24_team001/.condor/DataSet_temp/val_clean.txt'
clean_annotation_file(input_ann_file, output_ann_file)

from mmaction.registry import DATASETS

if 'DatasetZelda' in DATASETS.module_dict:
    print("DatasetZelda is registered.")
else:
    print("DatasetZelda is not registered.")


# Define dataset configurations
train_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

val_pipeline_cfg = [
    dict(type='VideoInit'),
    dict(type='VideoSample', clip_len=16, num_clips=5, test_mode=True),
    dict(type='VideoDecode'),
    dict(type='VideoResize', r_size=256),
    dict(type='VideoCrop', c_size=224),
    dict(type='VideoFormat'),
    dict(type='VideoPack')
]

train_dataset_cfg = dict(
    type='DatasetZelda',
    ann_file='/home/mler24_team001/.condor/DataSet_temp/train_clean.txt',
    pipeline=train_pipeline_cfg,
    data_root='/home/mler24_team001/.condor/DataSet_temp/',
    data_prefix=dict(video='train')
)

val_dataset_cfg = dict(
    type='DatasetZelda',
    ann_file='/home/mler24_team001/.condor/DataSet_temp/val_clean.txt',
    pipeline=val_pipeline_cfg,
    data_root='/home/mler24_team001/.condor/DataSet_temp/',
    data_prefix=dict(video='val')
)

print('Shape of the inputs: ', inputs.shape)
print('Image shape: ', data_sample.img_shape)
print('Number of clips: ', data_sample.num_clips)
print('Clip length: ', data_sample.clip_len)
print('Label: ', data_sample.gt_label)

BATCH_SIZE = 2

# Update dataloader configuration to include dataset instances directly
train_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=DATASETS.build(train_dataset_cfg)
)

val_dataloader_cfg = dict(
    batch_size=BATCH_SIZE,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=DATASETS.build(val_dataset_cfg)
)

# Build the dataloaders
train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
val_data_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)

# Test loading a batch of samples
batched_packed_results = next(iter(train_data_loader))

batched_inputs = batched_packed_results['inputs']
batched_data_sample = batched_packed_results['data_samples']

assert len(batched_inputs) == BATCH_SIZE
assert len(batched_data_sample) == BATCH_SIZE

print('Batch inputs shape: ', [inp.shape for inp in batched_inputs])
print('Batch data sample shapes: ', [ds.img_shape for ds in batched_data_sample])
print('Batch labels: ', [ds.gt_label for ds in batched_data_sample])