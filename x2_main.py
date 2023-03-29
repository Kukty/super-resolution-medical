import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import numpy as np
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from tqdm import tqdm

#Load model:
def Build_model(model_path,netscale=4):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
    upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=not False,
            gpu_id=None)
    return upsampler

def Execute(model_path,output_path,input,out_scale):
    upsampler = Build_model(model_path)
    if os.path.isfile(input):
        paths = [input]
    else:
        paths = sorted(glob.glob(os.path.join(input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            output, _ = upsampler.enhance(img, outscale=out_scale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f'{imgname}_out.jpg')
        cv2.imwrite(save_path, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-m',
        '--model_path',
        type=str,
        default='best.pth',
        help=('Model_path'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    args = parser.parse_args()
    Execute(args.model_path,args.output,args.input,args.outscale)
    print('OVER')


if __name__ == '__main__':
    main()
