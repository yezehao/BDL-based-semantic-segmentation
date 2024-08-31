import argparse
import logging
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from tqdm import tqdm

# Model Modules
from model.UNet import Unet
from model.r2unet import R2U_Net
from model.segnet import SegNet

from dataset import *
from metrics import *
from torchvision.transforms import transforms
from torchvision.models import vgg16


# Args
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=21)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='segnet',
                       help='UNet/myChannelUnet/segnet/r2unet')
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument('--dataset', default='MaSTr1325', 
                       help='dataset name:MaSTr1325')
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
            level=logging.INFO,
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
        train_dataset = MaSTr1325Dataset('train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = data.DataLoader(train_dataset, batch_size=args.batch_size)
        val_dataset = MaSTr1325Dataset("val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = data.DataLoader(val_dataset, batch_size=1)
        test_dataset = MaSTr1325Dataset("test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = data.DataLoader(test_dataset, batch_size=1)
    return train_dataloaders,val_dataloaders,test_dataloaders

def val(model, best_iou, val_dataloaders):
    model= model.eval()
    with torch.no_grad():
        i=0 
        Prs, Res, F1s = 0, 0, 0
        miou_total, dice_total = 0, 0
        num = len(val_dataloaders)  # validation dataset length
        #print(num)
        with tqdm(total=num, desc="Validation", unit="batch") as pbar:
            for x, _,pic,mask in val_dataloaders:
                x = x.to(device)
                y = model(x)
                # print(f"Validation: /val_mask/{os.path.basename(mask[0])}")
    
                # IOU & Dice Coeffeciency
                img_y = torch.squeeze(y).cpu().numpy()  # convert into numpy
                miou_total += get_iou(mask[0],img_y) 
                dice_total += get_dice(mask[0],img_y)
                # Precision / Recall / F1
                Pr, Re, F1 = get_precision(mask[0],img_y)
                Prs, Res, F1s = Prs + Pr, Res + Re, F1s + F1
                # Increment progress bar by one step
                pbar.update(1)    
                
        # Mean
        Prs = Prs/num
        Res = Res/num
        F1s = F1s/num
        aver_iou = miou_total/ num
        aver_dice = dice_total/num
        print('Precision=%.5f, Recall=%.5f, F1=%.5f' % (Prs,Res,F1s))
        print('Miou=%f,aver_dice=%f' % (aver_iou,aver_dice))
        logging.info('Precision=%.5f, Recall=%.5f, F1=%.5f' % (Prs,Res,F1s))
        logging.info('Miou=%f,aver_dice=%f' % (aver_iou,aver_dice))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_dice, Prs, Res, F1s
    
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
        best_iou,aver_iou,aver_dice, Prs, Res, F1s = val(model, best_iou, val_dataloader) # _ is mask
        metrics = {
            'aver_iou': aver_iou,
            'aver_dice': aver_dice,
            'Prs': Prs,
            'Res': Res, 
            'F1': F1s,
        }
        average_list.append(metrics)
        with open(f'./result/validation-epoch{args.epoch}.json', 'w') as f:
            json.dump(average_list, f, indent=4)
                
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    return model

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = f'./result/testing/{str(args.arch)}_{str(args.batch_size)}_{str(args.epoch)}_{str(args.dataset)}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    model.load_state_dict(torch.load(r'./result/saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    with torch.no_grad():
        i=0 
        num = len(val_dataloaders) 
        for pic,_,pic_path,mask_path in test_dataloaders:
            img_number = os.path.basename(mask_path[0])
            img_number = img_number[:-5]
            pic = pic.to(device)
            prediction = model(pic)
            prediction = torch.squeeze(prediction).cpu().numpy()

            dataset="MaSTr1325"
            root = f"../Dataset/{dataset}"
            test_show(root, threshold=0.2, img_number=img_number, prediction=prediction, dataset=dataset)

            if i < num:i+=1 


if __name__ =="__main__":
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