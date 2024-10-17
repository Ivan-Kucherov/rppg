import glob
import os
import re
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from rPPG_Toolbox.dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

"""The Base Class for data-loading.

Provides a pytorch-style data-loader for end-to-end training pipelines.
Extend the class to support specific datasets.
Dataset already supported: UBFC-rPPG, PURE, SCAMPS, BP4D+, and UBFC-PHYS.

"""
import csv
import glob
import os
import re
from math import ceil
from scipy import signal
from scipy import sparse
from scipy.interpolate import splrep,splev
from unsupervised_methods.methods import POS_WANG
from unsupervised_methods import utils
import math
from multiprocessing import Pool, Process, Value, Array, Manager

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from retinaface import RetinaFace   # Source code: https://github.com/serengil/retinaface



class McdLoader(BaseLoader):
    """Based on UBFC-rPPG"""

    def __init__(self, name, data_path, config_data):
        """Initializes an UBFC-rPPG dataloader.

            Args:

                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- ppg
                     |   |-- video
                     |   |-- db.csv
                -----------------

                name(string): name of the dataloader.

                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        df = glob.glob(data_path + os.sep + 'db.csv')
        if len(df) == 0 : 
            raise ValueError(self.dataset_name + " data paths empty!")
        df = pd.read_csv(glob.glob(data_path + os.sep + 'db.csv'))
        video = df[['video']].apply(lambda x: glob.glob(data_path + os.sep + x )[0])
        ppg = df[['ppg']].apply(lambda x: glob.glob(data_path + os.sep + x )[0])
        return df,video,ppg

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:  # return the full directory if begin == 0 and end == 1
            return data_dirs
        df,video,ppg = data_dirs
        file_num = len(data_dirs[0])
        strat,stop = int(begin * file_num), int(end * file_num)
        data_dirs_new = (df[strat:stop],video[strat:stop],ppg[strat:stop])
        return data_dirs_new
    
    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=8,multi_process = True):
        """Allocate dataset preprocessing across multiple processes.

        Args:
            data_dirs(List[str]): a list of video_files.
            config_preprocess(Dict): a dictionary of preprocessing configurations
            multi_process_quota(Int): max number of sub-processes to spawn for multiprocessing
        Returns:
            file_list_dict(Dict): Dictionary containing information regarding processed data ( path names)
        """
        print('Preprocessing dataset...')
        file_num = len(data_dirs[0])
        choose_range = range(0, file_num)
        pbar = tqdm(list(choose_range))

        # shared data resource
        manager = Manager()  # multi-process manager
        file_list_dict = manager.dict()  # dictionary for all processes to store processed files
        if multi_process:
            p_list = []  # list of processes
            running_num = 0  # number of running processes

            # in range of number of files to process
            for i in choose_range:
                process_flag = True
                while process_flag:  # ensure that every i creates a process
                    if running_num < multi_process_quota:  # in case of too many processes
                        # send data to be preprocessing task
                        p = Process(target=self.preprocess_dataset_subprocess, 
                                    args=(data_dirs,config_preprocess, i, file_list_dict))
                        p.start()
                        p_list.append(p)
                        running_num += 1
                        process_flag = False
                    for p_ in p_list:
                        if not p_.is_alive():
                            p_list.remove(p_)
                            p_.join()
                            running_num -= 1
                            pbar.update(1)
            # join all processes
            for p_ in p_list:
                p_.join()
                pbar.update(1)
            pbar.close()
        else:
            for i in choose_range:
                process_flag = True
                self.preprocess_dataset_subprocess(data_dirs,config_preprocess, i, file_list_dict)
            pbar.update(1)
        print('Preprocessing dataset complited')
        return file_list_dict

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ invoked by preprocess_dataset for multi_process."""
        df,video,ppg = data_dirs
        saved_filename = df[i]+'_'+df[i]['patient_id']
 
        # Read Frames
        if 'None' in config_preprocess.DATA_AUG:
            # Utilize dataset-specific function to read video
            frames = self.read_video(
                os.path.join(video[i]))
        elif 'Motion' in config_preprocess.DATA_AUG:
            # Utilize general function to read video in .npy format
            frames = self.read_npy_video(
                glob.glob(os.path.join(video[i])))
        else:
            raise ValueError(f'Unsupported DATA_AUG specified for {self.dataset_name} dataset! Received {config_preprocess.DATA_AUG}.')
        len = frames/float(self.config_data.FS)
        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            bvps = self.read_wave(
                os.path.join(ppg[i]), fs=self.config_data.FS,len=len)
            
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file,fs=30,bvp_fs= 100,len = None):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            str1 = f.read()
            str1 = str1.split("\n")[:-1]
            bvp = [float(x.split()[0]) for x in str1]
            if len(str1[0].split())==3:
                inter = True           
            else:
                inter = False
        if inter:
            inter = splrep(np.linspace(0, len, 100),k=3)
            bvp = splev(np.linspace(0, len, 30),inter)
        return np.asarray(bvp)
