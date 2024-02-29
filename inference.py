import os, sys
import argparse
import imageio.v2 as im
import cv2
import numpy as np
import torch

from irradiance_blur import irradiance_blur

def parse_arguments(args):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    #Paths
    parser.add_argument('--path', type=str, default="./")
    parser.add_argument('--file', type=str, default="inputs/")
    parser.add_argument('--save_path', type=str, default="outputs/")
    #Irradiance blur sattings
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--e', type=int, default=1)
    #Image size settings
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--colour', type=int, default=3)
    #Output settings
    parser.add_argument('--visible_output', type=str2bool, default=False)

    return parser.parse_known_args()

def gamma_correct_torch(img, gamma=2.2, alpha=1, inverse=0, quantile=0.5):
    clamped = torch.clamp(img,min=0)
    if inverse:
        return torch.pow((alpha*clamped),gamma)
    else:
        gam_img = torch.pow(clamped,(1/gamma))
        if alpha == 0:
            alpha = 2 * (torch.quantile(gam_img, quantile) + 1e-10)
        return (1/alpha)*gam_img

def get_file_list(img_dir):
    listOfFile = os.listdir(img_dir)
    allFiles = list()

    for entry in listOfFile:
        fullPath = os.path.join(img_dir, entry)

        if os.path.isdir(fullPath):
            allFiles = allFiles + get_file_list(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device: ', device, ":", torch.cuda.current_device())

    args, unknown = parse_arguments(sys.argv)

    args.file = args.path + args.file
    args.save_path = args.path + args.save_path

    if os.path.isdir(args.file):
        img_list = get_file_list(args.file)
        batch_size = args.batch_size
    else:
        img_list = [args.file]
        batch_size = 1

    blur_func = irradiance_blur((batch_size,args.colour,args.height,args.width), device)

    #declare batch [B,C,H,W]
    img_batch = torch.zeros((batch_size,args.colour,args.height,args.width),dtype=torch.float,device=device)
    batch_list = []
    num_batches = 0
    for i, img_name in enumerate(img_list):
        batch_list.append(img_name)
        cur_index = i-(num_batches*batch_size)

        img_filename, img_extension = os.path.splitext(os.path.basename(img_name))

        if img_extension == ".exr":
            img = im.imread(img_name, 'EXR-FI')

        elif img_extension == ".jpg" or img_extension == ".png":
            img = im.imread(img_name)

        img = cv2.resize(img, (args.width,args.height), interpolation=cv2.INTER_CUBIC)
        if args.colour == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_batch[cur_index] = torch.Tensor(img).to(device).permute((2,0,1))

        #process batch when it is full
        if cur_index == batch_size-1 or i+1 == len(img_list):
            num_batches +=1
            #get env_maps
            env_maps = blur_func(img_batch, args.alpha, args.e).permute((0,2,3,1))
            #save to dir
            for b, img_name in enumerate(batch_list):
                img_filename, img_extension = os.path.splitext(os.path.basename(img_name))
                if args.visible_output:
                    toned_env_map = torch.clip(gamma_correct_torch(env_maps[b].squeeze(0),alpha=0),0,1)*255.0
                    im.imwrite(args.save_path + img_filename + f"_{args.alpha}_{args.e}.png", toned_env_map.cpu().detach().numpy().astype('uint8'))
                else:
                    im.imwrite(args.save_path + img_filename + f"_{args.alpha}_{args.e}" + img_extension, env_maps[b].squeeze(0).cpu().detach().numpy())
               
            batch_list = []
