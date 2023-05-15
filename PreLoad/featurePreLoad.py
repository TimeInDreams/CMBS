import torch
import torchvision.transforms as transforms
import torchvision.models.vgg
import torch.nn.functional as F
import torchvision.models as models
from torchvggish import vggish, vggish_input
import cv2
import numpy as np
import numpy as np
import librosa
from python_speech_features import mfcc
import os
import PreLoad.visual_feature_extractor as vfe
import PreLoad.audio_feature_extractor as afe
import moviepy
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence
import shutil
import time

name_src_video='HeliTest.mp4'
video_path = os.path.join('.\PreLoad\Video\Src',name_src_video)
temp_path='.\PreLoad\Video\Temp'
output_folder = '.\PreLoad\Video\Opt'
video_opt_path=os.path.join(output_folder,'VideoSegment')
audio_opt_path=os.path.join(output_folder,'AudioSegment')

def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
def ExtractVideoSegment():
    segment_duration = 10  # in seconds
    


    lis = os.listdir(output_folder)
    # print(lis)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print('fps:',fps,'  frame_count:',frame_count)
    duration = frame_count / fps
    segments = int(duration / segment_duration) + 1

    for i in range(segments):
        start_frame = int(i * segment_duration * fps)
        end_frame = int(min((i + 1) * segment_duration * fps, frame_count))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        output_file = os.path.join(video_opt_path,name_src_video + f'_segment{i}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()

def ExtractAudioSegment():
    

    
    temp_audio_path=os.path.join(temp_path,'audio.wav')


    video = VideoFileClip(video_path)
    audio = video.audio
    if(video.audio==None):
        print('这段源视频',video_path,'中没有音频存在')
    # audio.write_audiofile('audio.wav')

    try:
        audio.write_audiofile(temp_audio_path)
    except AttributeError as e:
        print('Error:', e)


    audio = AudioSegment.from_wav(temp_audio_path)
    length = len(audio)
    ten_seconds = 10 * 1000 # ten seconds in milliseconds
    for i in range(0, length, ten_seconds):
        chunk = audio[i:i+ten_seconds]
        # chunk.export(f"chunk{i}.wav", format="wav")
        chunk.export(os.path.join(audio_opt_path,name_src_video+f"_chunk{i}.wav"), format="wav")


def PreLoad():
    #清空文件夹
    del_file(video_opt_path)
    del_file(audio_opt_path)
    #将视频切片，得到每个十秒的视频段和每个十秒的音频段
    ExtractVideoSegment()
    ExtractAudioSegment()

    T1 = time.perf_counter()
    video_features =vfe.video_feature_extract(video_opt_path)
    audio_features =afe.audio_feature_extract(audio_opt_path)
    T2 = time.perf_counter()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))

    return video_features,audio_features


def main():
    PreLoad()
if __name__ == '__main__':
    main()