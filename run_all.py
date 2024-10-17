import sys
import yaml
import glob
import os
from tqdm import tqdm

def change(data):
    data['TEST']['DATA']['DO_PREPROCESS'] = True

    data['TEST']['DATA']['DATA_PATH'] = 'E:/FirstProj/mcd_rppg'
    data['TEST']['DATA']['CACHED_PATH'] = './data/Custom_cached'

    data['TEST']['DATA']['BEGIN'] = 0.0
    data['TEST']['DATA']['END'] = 0.001

    data['TEST']['DATA']['PREPROCESS']['CROP_FACE']['BACKEND'] = 'MP'
    return data
if __name__ == "__main__":
    for i in tqdm(glob.glob('./CustomConfigs/*.yaml')):
        with open(i) as fp:
            data = yaml.load(fp,Loader=yaml.FullLoader)
            data = change(data)
            fp.close()
        with open(i , "w") as f:
            yaml.dump(data, f)
        os.system(f'python main.py --config_file {i}')