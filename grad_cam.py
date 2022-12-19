import os
import tqdm
from torch.utils.data import DataLoader
import argparse
import torchvision.models as models
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM
import torch.nn as nn
from model import *

from dataset import Crop_data

data_dict = {0: 'asparagus', 1: 'bambooshoots', 2: 'betel', 3: 'broccoli', 4: 'cauliflower', 5: 'chinesecabbage', 6: 'chinesechives', 7: 'custardapple', 8: 'grape', 9: 'greenhouse', 10: 'greenonion', 11: 'kale', 12: 'lemon', 13: 'lettuce', 14: 'litchi', 15: 'longan', 16: 'loofah', 17: 'mango', 18: 'onion', 19: 'others', 20: 'papaya', 21: 'passionfruit', 22: 'pear', 23: 'pennisetum', 24: 'redbeans', 25: 'roseapple', 26: 'sesbania', 27: 'soybeans', 28: 'sunhemp', 29: 'sweetpotato', 30: 'taro', 31: 'tea', 32: 'waterbamboo'}


def grad_cam(input_tensor, model):
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True)

    return grayscale_cam



def visualization(opt, model, test_loader):
    model.eval()
    for image, label, file_names in tqdm.tqdm(test_loader):
        image = image.cuda()
        pred = model(image) #(B, num_classes)
        pred_label = pred.cpu().detach().numpy()
        pred_label = np.argmax(pred_label, axis=1)

        grayscale_cam = grad_cam(image, model)
        for i, gray_cam_image in enumerate(grayscale_cam):
            class_name = data_dict[label[i].item()]
            pred_name = data_dict[pred_label[i]]

            # if pred_name != class_name:
            file_path = os.path.join(opt.root, class_name, file_names[i])
            try:
                image_bgr = cv2.imread(file_path)
                h, w, c= image_bgr.shape
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_ori = image_rgb.copy()

                image_rgb = (image_rgb / 255.0).astype(np.float32)
                gray_cam_image = cv2.resize(gray_cam_image, (w, h))
                visualization = show_cam_on_image(image_rgb, gray_cam_image, use_rgb=True)
                # plt.figure(figsize=(15,15))
                plt.subplot(121)
                plt.title(pred_name)
                plt.imshow(visualization)
                plt.subplot(122)
                plt.title(class_name)
                plt.imshow(image_ori)
                plt.savefig(os.path.join(opt.out_path, file_names[i]))
                print("predict error: %s \t predict: %s \t true_label: %s" % (file_names[i], pred_name, class_name))
                # plt.show()
            except:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/root/M11115Q13/dataset/image_size1024', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=33, help='number of classes')
    parser.add_argument('--mode', type=str, default="valid", help='valid/test')
    parser.add_argument('--five_crop', type=bool, default=False, help='whether to use five_crop')
    parser.add_argument('--out_path', type=str, default="./output_gradcam")

    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")

    parser.add_argument('--model', default='resnext50', help='resnet18/resnet50')
    parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu workers')
    parser.add_argument('--load', default='/root/M11115Q13/crop_classification/checkpoints/resnext50_MLP/model_epoch9_acc0.8644.pth', help='path to model to continue training')
    
    opt = parser.parse_args()
    os.makedirs(opt.out_path, exist_ok=True)

    test_data = Crop_data(opt, opt.root, opt.mode)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)

    #model = models.resnet18(pretrained=True)
    if opt.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, opt.num_classes)
    elif opt.model == 'convnext_small':
        model = convnext_small(opt)
    elif opt.model == 'convnext_base':
        model = convnext_base(opt)
    model.fc = nn.Linear(model.fc.in_features, opt.num_classes)
    if opt.load != '':
        print(f'loading pretrained model from {opt.load}')
        load = torch.load(opt.load)
        load = {k.replace('module.', ''):v for k, v in load.items()}
        model.load_state_dict(load)
    model = model.cuda()
    model.eval()

    visualization(opt, model, test_loader)