# %%
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

# %%
def PSNR(im_gt,im_output):
    mse = np.mean((im_gt-im_output)**2)
    if mse ==0:
        psnr = float('inf')
    else:
        psnr = 10* np.log10(255.0**2/mse)
    return psnr

# %%
import os
os.chdir('/data/public/PATH-DT-MSU/SR_patches_10x_20x_40x/')
netscale = 4
output_scale = 2

# %%
iter_list = list(range(80000, 80001, 5000))
model_name_list = ['net_g_'+ str(model_name) +'.pth' for model_name in iter_list]
# iter_list
# print(model_name_list)
model_dir = '/home/zsun/Forme/Real-ESRGAN-master-2/experiments/finetune_RealESRGANx4plus_400k_pairdata/models'
# model_dir = '/home/zsun/Forme/Real-ESRGAN-master-2/fine-tune_1/models'
mean_psnr_list = np.zeros_like(iter_list,dtype=np.float32)
print(mean_psnr_list)

# %%
info_path = '/home/zsun/Forme/Real-ESRGAN-master-2/meta_test_double.txt'
output_path = '/home/zsun/Forme/Real-ESRGAN-master-2/outputs_80000_3full'
with open(info_path,'r') as f :
    i = 0
    img_paths = [line.strip() for line in f]
    for model_name in model_name_list:
        model_path = os.path.join(model_dir,model_name)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=not False,
            gpu_id=0)
        print(f"model {model_name} is created")
        sum_psnr = 0

        for pair_path in tqdm(img_paths,total = len(img_paths)):
            ref_path = pair_path.split(', ')[0]
            lr_path = pair_path.split(', ')[1]
            # print(ref_path,lr_path)
            im_ref = cv2.imread(ref_path)
            im_lr = cv2.imread(lr_path)
            try:
                im_output,_ = upsampler.enhance(im_lr,outscale=output_scale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            psnr = PSNR(im_gt=im_ref,im_output=im_output)
            os.makedirs(output_path, exist_ok=True)
            lr = lr_path.split('/')[0]+lr_path.split('/')[2]
            save_path = os.path.join(output_path,f'{lr}_out.png')
            # print(save_path)
            cv2.imwrite(save_path,im_output)
            # print(psnr)
            sum_psnr +=psnr
            # break
        print(sum_psnr)
        mean_psnr_list[i] = sum_psnr / len(img_paths)
        print(mean_psnr_list[i])
        i = i+1
# net_g
#5000  29.601673
#10000 29.631217
#15000 29.64325
#20000 29.602331
#25000 29.598352
#30000 29.596682
#35000 29.591452
#40000 29.596299
# np.savez('Full_res_test_3',mean_psnr_list)
print("OVER")

# %%



