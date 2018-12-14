import torch

from imageio import imread, imsave
from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from models import DispNetS, DispResNet
from utils import tensor2array

parser = argparse.ArgumentParser(description='Inference script for DispNet learned with \
                                 Structure from Motion Learner inference on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--output-disp", action='store_true', help="save disparity img")
parser.add_argument("--output-depth", action='store_true', help="save depth img")
parser.add_argument("--pretrained", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=128, type=int, help="Image height")
parser.add_argument("--img-width", default=416, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--dataset-dir", default='E:/PCProjects/val_image', type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='E:/PCProjects/output', type=str, help="Output directory")

parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    args = parser.parse_args()
    if not(args.output_disp or args.output_depth):
        print('You must at least output one value !')
        return

    # disp_net = DispNetS().to(device)
    disp_net = DispResNet(3).to(device)
    weights = torch.load(args.pretrained)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_disp = output_dir / 'disp'
    output_depth = output_dir / 'depth'
    output_disp.makedirs_p()
    output_depth.makedirs_p()

    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = [dataset_dir/file for file in f.read().splitlines()]
    else:
        test_files = sum([dataset_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])

    print('{} files to test'.format(len(test_files)))

    for file in tqdm(test_files, ncols=100):

        img = imread(file).astype(np.float32)

        h,w,_ = img.shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        tensor_img = torch.from_numpy(img).unsqueeze(0)
        tensor_img = ((tensor_img/255 - 0.5)/0.2).to(device)

        output = disp_net(tensor_img)[0][0]

        if args.output_disp:
            disp = (255*tensor2array(output, max_value=None, colormap='bone', channel_first=False)).astype(np.uint8)
            imsave(output_disp/'{}_disp{}'.format(file.namebase,file.ext), disp)
        if args.output_depth:
            depth = 1/output
            depth = (255*tensor2array(depth, max_value=10, colormap='rainbow', channel_first=False)).astype(np.uint8)
            imsave(output_depth/'{}_depth{}'.format(file.namebase,file.ext), depth)


if __name__ == '__main__':
    main()