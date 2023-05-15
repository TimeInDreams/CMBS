import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np


from configs.opts import parser
from model.main_model import supv_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset
import torch.nn.functional as F

import PreLoad.featurePreLoad as preload
# 分类每个十秒视频段，并将分类结果输出到result
dict_event={
    0:"Church_bell" ,
    1:"Man_speaking",
    2:" Dog barking",
    3:" Airplane",
    4:" Racing car",
    5:"Women speaking",
    6:"  Helicopter ",
    7:"Violin",
    8:"Flute",
    9:"Ukutele",
    10:" Frying food",
    11:" Truck"
}
def get_classification_video_segment(is_event_scores, event_scores):
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5
    scores_mask = scores_pos_ind == 0
    _, event_class = event_scores.max(-1)  # foreground classification，前景分类,对整个十秒的事件判定
    pred = scores_pos_ind.long()
    
    
    if(pred.shape==torch.Size([10])):
        pred=pred.expand(1,-1)
        scores_mask=scores_mask.expand(1,-1)
    pred *= event_class[:, None]

    pred[scores_mask] = 28

    idx=event_class.item
    result.append(dict_event[idx])
    print('pred:',pred.shape)
    print('该十秒视频事件分类为',event_class)
    print('整个十秒视频的详细推理结果：',pred)
def get_result_video_class():
    value_cnt = {}  # 将结果用一个字典存储
    # 统计结果
    for value in result:
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt[value] = value_cnt.get(value, 0) + 1

    # 打印输出结果
    print(value_cnt)
    print([key for key in value_cnt.keys()])
    print([value for value in value_cnt.values()])


result=[]
def main():
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    '''model setting'''
    config_path = 'configs/main.json'
    with open(config_path) as fp:
        config = json.load(fp)
    print(config)
    mainModel = main_model(config['model'])
    mainModel = nn.DataParallel(mainModel).cuda()
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=0.001)
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    '''Resume from a checkpoint'''
    resume_path='.\model_sup\model_epoch_23_top1_77.338_task_Supervised_best_model.pth.tar'
    mainModel.load_state_dict(torch.load(resume_path))

    mainModel.eval()
    mainModel.double()



    video_features,audio_features=preload.PreLoad()
    video_features=torch.from_numpy(video_features).double()
    audio_features=torch.from_numpy(audio_features).double()

    video_dir='.\PreLoad\Video\Opt\VideoSegment'
    lis = os.listdir(video_dir)
    len_video_dir = len(lis)
    # video_features = np.zeros([len_data, 10, 7, 7, 512]) # 10s long video

    audio_dir='.\PreLoad\Video\Opt\AudioSegment'
    lis = os.listdir(audio_dir)
    len_audio_dir = len(lis)
    # audio_features = np.zeros([len_data, 10, 128])
    
    len_data = len_audio_dir-1
    if(len_audio_dir!=len_video_dir):
        print('这个视频的音频分段数和视频分段数不同')


    print('video_feature_dtype:',video_features[0].dtype)
    
    print('开始处理特征:')
    for i in range(len_data):
        vf=video_features[i].unsqueeze(0).double()
        af=audio_features[i].unsqueeze(0).double()
        print('video_feature_dtype:',video_features[0].dtype)
        print(f'以下为第{i}个十秒片段的推理结果:')
        T1 = time.perf_counter()
        with torch.no_grad():
            is_event_scores, event_scores, audio_visual_gate, _ = mainModel(vf, af)
        T2 = time.perf_counter()
        print('模型推理一轮运行时间:%s毫秒' % ((T2 - T1)*1000))
        print("is_event_scores:",is_event_scores.shape)
        print("event_scores:",event_scores.shape)
        print("audio_visual_gate:",audio_visual_gate.shape)
        get_classification_video_segment(is_event_scores,event_scores)

        
if __name__ == '__main__':
    main()