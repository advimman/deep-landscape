import os
import traceback
import argparse
from collections import OrderedDict

import torch
from tqdm import tqdm

import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', default="configs/test/test.yaml", type=str, help='Path to options YAML file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs([opt['path']['root']])

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        for test_data in tqdm(test_loader, total=len(test_loader)):
            if len(test_data) == 0:
                continue
            try:
                img_dir = os.path.join(opt['path']['root'], f"{test_data['video_name'][0]}")
                util.mkdir(img_dir)
                save_img_path = os.path.join(img_dir, f'{test_data["frame_name"][0]}.jpg') #img_name
                if os.path.exists(save_img_path):
                    print(f"Skipping {test_data['frame_name'][0]}.jpg - because already exists in {save_img_path}")
                    continue

                model.feed_data(test_data, need_GT=False)
                model.test()

                visuals = model.get_current_visuals(need_GT=False)
                sr_img = util.tensor2img(visuals['SR']) 
                util.save_img(sr_img, save_img_path)

            except Exception as e:
                print(f"Exception {e}: {traceback.format_exc()}")
                pass