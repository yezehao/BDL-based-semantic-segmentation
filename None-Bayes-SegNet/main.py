'''
author:zhujunwen
Guangdong University of Technology
'''
import argparse
import logging
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim

# Model Modules
from model.UNet import Unet
from model.r2unet import R2U_Net
from model.segnet import SegNet

from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from torchvision.models import vgg16

# Args
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='resnet34_unet',
                       help='UNet/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='driveEye',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung')
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold",type=float,default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,f"{args.arch}_bsize{args.batch_size}_{args.dataset}_epoch{args.epoch}")
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = Unet(3, 3).to(device)
    if args.arch == 'segnet':
        model = SegNet(3,3).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3,3).to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset == 'MaSTr1325':
        train_dataset = MaSTr1325Dataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = MaSTr1325Dataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = data.DataLoader(val_dataset, batch_size=1)
        test_dataset = MaSTr1325Dataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = data.DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model, best_iou, val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0 
        miou_total = 0
        dice_total = 0
        num = len(val_dataloaders)  # validation dataset length
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  # convert into numpy

            print(f"The validation mask: /val_mask/{os.path.basename(mask[0])}")
            miou_total += get_iou(mask[0],img_y) 
            dice_total += get_dice(mask[0],img_y)
            if i < num:i+=1   
                
        aver_iou = miou_total / num
        aver_dice = dice_total/num
        print('Miou=%f,aver_dice=%f' % (aver_iou,aver_dice))
        logging.info('Miou=%f,aver_dice=%f' % (aver_iou,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_dice #,aver_hd
    
def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    average_list = []
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)
                loss = criterion(output, labels)
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)

        best_iou = 0
        best_iou,aver_iou,aver_dice = val(model, best_iou, val_dataloader) # _ is mask
        metrics = {
            'aver_iou': aver_iou,
            'aver_dice': aver_dice,
        }
        average_list.append(metrics)
        with open('average.json', 'w') as f:
            json.dump(average_list, f, indent=4)
                
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    return model

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = f'./saved_predict/{str(args.arch)}_{str(args.batch_size)}_{str(args.epoch)}_{str(args.dataset)}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0 
        num = len(val_dataloaders) 
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  # convert into numpy

            fig = plt.figure()
            ax1 = fig.add_subplot(2, 3, 1)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]))
            #print(pic_path[0])
            ax2 = fig.add_subplot(2, 3, 2)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(2, 3, 4)
            ax3.set_title('mask_0')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            ax4 = fig.add_subplot(2, 3, 5)
            ax4.set_title('mask_1')
            plt.imshow(Image.open(mask_path[1]), cmap='Greys_r')            
            ax5 = fig.add_subplot(2, 3, 6)
            ax5.set_title('mask_2')
            plt.imshow(Image.open(mask_path[2]), cmap='Greys_r')
            #print(mask_path[0])
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.tif'
                    plt.savefig(saved_predict)
                else:
                    plt.savefig(dir +'/'+ mask_path[0].split('\\')[-1])
            #plt.pause(0.01)

            if i < num:i+=1   #处理验证集下一张图
        plt.show()


if __name__ =="__main__":
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
    ])
    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
        ])

    # mask tranfrom to tensor
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The Device: {device}")
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('Models:%s,\nEpoch:%s,\nBatch Size:%s\nDataset:%s' % \
            (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
            (args.arch, args.epoch, args.batch_size,args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders,val_dataloaders,test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)