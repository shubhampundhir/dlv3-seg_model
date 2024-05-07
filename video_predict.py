from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics
import cv2

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 14
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    
    img_folders=[
        "Jamulni-Adumber",
    "Khedi-Paratwadaroad-Malegaon",
    "L063-Masod-BisnoorRoadtoSahangaon", "MRL07-AnaimalaitokaliyapuramVeparaipathi",
         "T01-HariyanicheruvutoGantimarriVia.Kuntimaddi",
   "T02-Amla-Tirmau-KharpadakheditoBahmniPankhaMultaiBordehiRoad",
    "T02-DausatoBhandanaviaGupteshwarSingwaraMalarana",
     "T13-JhanpdatoNH23ViaDaulatpura",
   "T25-MDR153BidarkhatoNH148ViaNayagaon,Kolyawas,Pyariwas",  
   "T02-Gowdanahall-ChandrakacharlaroadtoTDPalli", 
 "L026-KolarpattyChettipalayamRoadtoKoolanaickenpattyRoad" ,   "L022-LaxmiNagarVR",
    "L064-HangalgundtoNagam-SoafShaliShitru",
    "MRL14-PattanamKothavadiroad",
    "T16-Masod-Hiwarkhed-DhablatoSendurjana",
    "T07-BisnoortoJogikheda",
    "L131-SunhanitoDoon"
  ]
    for img_file in img_folders:
        input_path = os.path.join("/home/alive/Desktop/himanshi/mord/DeepLabV3Plus-Pytorch-mord/datasets/video_data/frames/Videos_from_the_emarg_portal", img_file)
        print(input_path)
        image_files = []
        
        if os.path.isdir(input_path):
            for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(input_path, f'**/*.{ext}'), recursive=True)
                if len(files) > 0:
                    image_files.extend(files)
    

        print(len(image_files))
            
        
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            print("Resume model from %s" % opts.ckpt)
            del checkpoint
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model.to(device)

        #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

        if opts.crop_val:
            transform = T.Compose([

                    T.Resize(opts.crop_size),
                    T.CenterCrop(opts.crop_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        else:
            transform = T.Compose([

                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                ])
        if opts.save_val_results_to is not None:
            os.makedirs(opts.save_val_results_to, exist_ok=True)
    

        with torch.no_grad():
            model = model.eval()
            for img_path in tqdm(image_files):
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext)-1]
                img = Image.open(img_path).convert('RGB')
                original_img = np.array(img)  # Convert PIL image to NumPy array

                img = transform(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)

                pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_preds = Image.fromarray(colorized_preds)
                original_h, original_w = original_img.shape[:2]

                # Check and resize prediction if needed
                pred_h, pred_w = colorized_preds.size
                if pred_h != original_h or pred_w != original_w:
                    resized_pred = colorized_preds.resize((original_w, original_h))  # Ensure resizing to original dimensions
                    colorized_preds = np.array(resized_pred)  # Convert back to NumPy array

                                # Superimpose the colorized_preds on the original image
                blended_img = cv2.addWeighted(original_img, 0.6, np.array(colorized_preds), 0.4, 0)

                output_folder = os.path.join(opts.save_val_results_to, img_file)
                # colored_folder = os.path.join(output_folder, "colored")
                overlay_folder = os.path.join(output_folder, "overlay")

                # Create folders if they don't exist
                # os.makedirs(colored_folder, exist_ok=True)
                os.makedirs(overlay_folder, exist_ok=True)

                # Save colored prediction
                # pred_path = os.path.join(colored_folder, img_name + '_color.png')
                # cv2.imwrite(pred_path, cv2.cvtColor(np.array(colorized_preds), cv2.COLOR_BGR2RGB))

                # Save overlayed image
                overlay_path = os.path.join(overlay_folder, img_name + '_overlay.png')
                cv2.imwrite(overlay_path, cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))

          
if __name__ == '__main__':
    main()