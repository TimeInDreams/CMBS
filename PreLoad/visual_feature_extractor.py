import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import cv2
import pylab
import imageio
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
# from keras.preprocessing import image
import keras.utils as image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    

    return num

def video_feature_extract(video_dir):
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output) # vgg pool5 features

    # path of your dataset
    # video_dir = '.\PreLoad\Video\Opt'
    lis = os.listdir(video_dir)

    len_data = len(lis)-1
    video_features = np.zeros([len_data, 10, 7, 7, 512]) # 10s long video
    t = 10 # length of video
    sample_num = 16 # frame number for each second

    c = 0
    for num in range(len_data):

        '''feature learning by VGG-net'''
        video_index = os.path.join(video_dir, lis[num]) # path of videos
        
        vid = imageio.get_reader(video_index, 'ffmpeg')
        
        vid_len = vid.count_frames()
        # print('vid_len',vid_len)
        frame_interval = int(vid_len / t)
        # print('frame_inter:',frame_interval)
        frame_idx_list = video_frame_sample(frame_interval, t, sample_num)
        print('frame_num:',len(frame_idx_list))
        imgs = []
        for i, im in enumerate(vid):
            x_im = cv2.resize(im, (224, 224))
            # print('after_resize_224_img:',x_im)
            imgs.append(x_im)
        vid.close()
        extract_frame = []
        for n in frame_idx_list:
            # print('n:',n)
            extract_frame.append(imgs[n])

        feature = np.zeros(([10, 16, 7, 7, 512]))
        for j in range(len(extract_frame)):
            y_im = extract_frame[j]

            x = image.img_to_array(y_im)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            pool_features = np.float32(model.predict(x))

            tt = int(j / sample_num)
            video_id = j - tt * sample_num
            feature[tt, video_id, :, :, :] = pool_features
        feature_vector = np.mean(feature, axis=(1)) # averaging features for 16 frames in each second
        print(feature_vector.shape)
        video_features[num, :, :, :, :] = feature_vector
        c += 1

    return video_features

# save the visual features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file
# with h5py.File('.../video_cnn_feature.h5', 'w') as hf:
#     hf.create_dataset("dataset", data=video_features)