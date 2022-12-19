import argparse
from dataset import Crop_data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import tqdm
import os

from torchstat import stat
from utils import get_acc, get_wp_f1, fixed_seed
from model import *
from model_component.RepLKNet import *
import torchvision.models as models
import torch.nn as nn

import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

import warnings
warnings.filterwarnings("ignore")

def train_batch(opt, model, optimizer, criterion, image, label):
    optimizer.zero_grad()

    if opt.model == 'cnn_pvt_fusion':
        out_CNN, out_PVT, pred = model(image)
        loss_CNN = criterion(out_CNN, label)
        loss_PVT = criterion(out_PVT, label)
        loss_all = criterion(pred, label)
        loss = 0.5 * loss_CNN + 0.5 * loss_PVT + loss_all
    else:
        pred = model(image)
        if opt.model == 'ViT' or opt.model == 'beit':
            pred = pred.logits

        loss = criterion(pred, label)

    loss.backward()
    optimizer.step()

    return loss, pred

def validation(opt, model, criterion, val_loader, writer, epoch):
    model.eval()

    y_true = torch.tensor([]).type(torch.int16)
    y_pred = torch.tensor([]).type(torch.int16)
    total_correct = 0
    total_label = 0
    val_loss = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label, _ in val_loader:
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()

            if opt.model == 'cnn_pvt_fusion':
                out_CNN, out_PVT, pred = model(image)
                loss_CNN = criterion(out_CNN, label)
                loss_PVT = criterion(out_PVT, label)
                loss_all = criterion(pred, label)
                loss = 0.5 * loss_CNN + 0.5 * loss_PVT + loss_all
            else:
                pred = model(image)
                if opt.model == 'ViT' or opt.model == 'beit':
                    pred = pred.logits

                loss = criterion(pred, label)

            correct, total = get_acc(pred, label)

            total_label += total
            total_correct += correct
            val_acc = (total_correct / total_label) * 100

            val_loss += loss

            label = label.cpu().detach()
            pred = pred.cpu().detach()
            y_true = torch.cat((y_true, label), 0)
            y_pred = torch.cat((y_pred, pred), 0)

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                Accuracy=f"{val_acc:.2f}"
            )
    
    f1_dict, WP_value = get_wp_f1(y_pred, y_true)

    pbar.set_postfix(
        loss=f"{val_loss:.4f}",
        Accuracy=f"{val_acc:.2f}",
        WP_value=f"{WP_value:.4f}"
    )
    pbar.close()
    
    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation accuracy', val_acc, epoch)
    writer.add_scalar('validation WP_value', WP_value, epoch)
    writer.add_scalars('validation f1_score per class', 
    {'asparagus':f1_dict[0], 'bambooshoots':f1_dict[1], 'betel':f1_dict[2], 'broccoli':f1_dict[3], 'cauliflower':f1_dict[4], 'chinesecabbage':f1_dict[5],
    'chinesechives':f1_dict[6], 'custardapple':f1_dict[7], 'grape':f1_dict[8], 'greenhouse':f1_dict[9], 'greenonion':f1_dict[10],
    'kale':f1_dict[11], 'lemon':f1_dict[12], 'lettuce':f1_dict[13], 'litchi': f1_dict[14], 'longan': f1_dict[15], 
    'loofah': f1_dict[16], 'mango': f1_dict[17], 'onion': f1_dict[18], 'others': f1_dict[19], 'papaya': f1_dict[20], 
    'passionfruit': f1_dict[21], 'pear': f1_dict[22], 'pennisetum': f1_dict[23], 'redbeans': f1_dict[24], 'roseapple': f1_dict[25], 
    'sesbania': f1_dict[26], 'soybeans': f1_dict[27], 'sunhemp': f1_dict[28], 'sweetpotato': f1_dict[29], 'taro': f1_dict[30], 'tea': f1_dict[31], 'waterbamboo': f1_dict[32]}, epoch)

    return WP_value

def main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader):
    writer = SummaryWriter('runs/%s_MLP' % opt.model)
    
    device = torch.device('cuda')
    
    criterion = criterion.cuda()
    model = model.to(device)

    """training"""
    print('Start training!')
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    max_WP = 0.
    train_update = 0

    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train(True)
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")

        total_loss = 0
        total_correct = 0
        total_label = 0

        for image, label, _ in train_loader:
            image, label = image.cuda(), label.cuda()
            train_loss, pred = train_batch(opt, model, optimizer, criterion, image, label)

            correct, total = get_acc(pred, label)

            total_label += total
            total_correct += correct
            acc = (total_correct / total_label) * 100

            total_loss += train_loss
        
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                Accuracy=f"{acc:.2f}%"
            )

            writer.add_scalar('training loss', train_loss, train_update)
            writer.add_scalar('training accuracy', acc, train_update)
            train_update += 1

        pbar.close()

        val_WP = validation(opt, model, criterion, val_loader, writer, epoch)
        if max_WP <= val_WP:
            print('save model!!')
            max_WP = val_WP
            torch.save(model.state_dict(), os.path.join(opt.save_model, opt.model+'_MLP', 'model_epoch%d_acc%.4f.pth' % (epoch, max_WP)))

        scheduler.step(val_WP)

        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    print('best WP:%.2f' % (max_WP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")

    parser.add_argument('--root', default='/root/M11115Q13/dataset/image_size1024', help='path to dataset')
    parser.add_argument('--five_crop', type=bool, default=False, help='whether to use five_crop')
    parser.add_argument('--num_classes', type=int, default=33, help='number of classes')

    parser.add_argument('--optimizer', default='sgd', help='adam/sgd')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--model', default='convnext_small', help='resnext50/convnext_small/convnext_base')

    parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu workers')
    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--save_model', default='./checkpoints', help='path to save model')
    
    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    fixed_seed(2022)
    os.makedirs(os.path.join(opt.save_model, opt.model+'_MLP'), exist_ok=True)

    train_data = Crop_data(opt, opt.root, 'train')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu, drop_last=True)
    val_data = Crop_data(opt, opt.root, 'valid')
    val_loader = DataLoader(val_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu)

    if opt.model == 'convnext_small':
        #model = models.convnext_small(pretrained=True)
        model = convnext_small(opt)
        for param in model.parameters():
            param.requires_grad = False
        model.model.head = nn.Linear(model.model.head.in_features, opt.num_classes)

    elif opt.model == 'convnext_base':
        #model = models.convnext_base()
        model = convnext_base(opt)
        for param in model.parameters():
            param.requires_grad = False
        model.model.head = nn.Linear(model.model.head.in_features, opt.num_classes)
            
    elif opt.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, opt.num_classes)


    elif opt.model == 'cnn_pvt_fusion':
        model = cnn_pvt_fusion(opt)

    if opt.load != '':
        print(f'loading pretrained model from {opt.load}')
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(opt.load))

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    if opt.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay = 5e-4)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay = 5e-4, momentum=0.9)
    
    """lr_scheduler"""
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5)

    #print( stat(model, (3,1024,1024)) )

    main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader)