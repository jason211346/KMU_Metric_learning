"""
    PyTorch Package for SoftTriple Loss

    Reference
    ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling"

    Copyright@Alibaba Group

"""

import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tqdm import tqdm 
# from loss.SoftTriple import SoftTriple
from loss.softtriple import SoftTriple
import evaluation as eva
# import net
# from net.bninception import bninception

from net.cnn_ensemble import cnn_ensemble
from dataset.dataset_kmu_cnn import Dataset
# from net.resnet_model import ResNet50
# from net.resnet_model_cls import ResNet50
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size')
parser.add_argument('--modellr', default=0.0001, type=float,
                    help='initial model learning rate')
parser.add_argument('--centerlr', default=0.01, type=float,
                    help='initial center learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    help='weight decay', dest='weight_decay')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--eps', default=0.01, type=float,
                    help='epsilon for Adam')
parser.add_argument('--rate', default=0.1, type=float,
                    help='decay rate')
parser.add_argument('--dim', default=512, type=int,
                    help='dimensionality of embeddings')
parser.add_argument('--freeze_BN', action='store_true',
                    help='freeze bn')
parser.add_argument('--la', default=20, type=float,
                    help='lambda')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--tau', default=0.2, type=float,
                    help='tau')
parser.add_argument('--margin', default=0.01, type=float,
                    help='margin')
parser.add_argument('-C', default=2, type=int,
                    help='C')
parser.add_argument('-K', default=10, type=int,
                    help='K')
parser.add_argument('--model_name', default='CUB_final.pth', type=str,
                    help='MN')

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = im.split()
    return Image.merge('RGB', (b, g, r))


def main():
    args = parser.parse_args()
    out_path = '/root/notebooks/nfs/work/jason.chen/project/triplet_loss/pytorch-triplet-loss/SoftTriple/out_model/'
    if not os.path.isdir(out_path+args.model_name):
        os.mkdir(out_path+args.model_name)
#     writer = SummaryWriter(os.path.join(out_path+args.model_name, "summary"))
    # create model
#     model = bninception(args.dim)
#     model = ResNet50(num_classes=args.dim)
    model = cnn_ensemble()
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = SoftTriple(args.la, args.gamma, args.tau, args.margin, args.dim, args.C, args.K).cuda()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.modellr},
                                  {"params": criterion.parameters(), "lr": args.centerlr}],
                                 eps=args.eps, weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # load data
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'val')
#     normalize = transforms.Normalize(mean=[104., 117., 128.],
#                                      std=[1., 1., 1.])

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.Lambda(RGB2BGR),
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.mul(255)),
#             normalize,
#         ]))

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.workers, pin_memory=True)

#     test_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(testdir, transforms.Compose([
#             transforms.Lambda(RGB2BGR),
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.mul(255)),
#             normalize,
#         ])),
#         batch_size=args.batch_size, shuffle=False,
#         num_workers=args.workers, pin_memory=True)

#     train_set,valid_set = dataload(traindir,testdir)
    
#     train_loader = torch.utils.data.DataLoader(train_set,batch_size=args.batch_size, shuffle=True,num_workers=4) 

#     test_loader = torch.utils.data.DataLoader(valid_set,batch_size=args.batch_size,shuffle=False,num_workers=4)

    train_dataset   = Dataset(traindir, shuffle_pairs=True, augment=True)
    val_dataset     = Dataset(testdir, shuffle_pairs=True, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=8, drop_last=True)
    test_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=6,num_workers=8, drop_last=True)
    
    
    val_loss_list=[]
    train_loss_list=[]
    val_acc_list=[]
    train_acc_list=[]
    best_val =100
    #--train
    for epoch in range(args.start_epoch, args.epochs):
        correct = 0
        total = 0
        running_loss = 0.0
        print('Training in Epoch[{}]'.format(epoch))
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc, train_loss = train(train_loader, correct,total,running_loss,model, criterion, optimizer, args)
        
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
#         writer.add_scalar('train_loss', train_loss, epoch)
#         writer.add_scalar('train_acc', train_acc, epoch)
        
    # evaluate on validation set
        correct = 0
        total = 0
        running_loss = 0.0
        nmi, recall ,val_acc , val_loss= validate(test_loader,correct,total,running_loss, model,criterion, args)
#         writer.add_scalar('val_loss', val_loss, epoch)
#         writer.add_scalar('val_acc', val_acc, epoch)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path+args.model_name, args.model_name+'_best.pth')
            )  
        
    #--save final model 
    plt.plot(val_loss_list, label='val_loss')
    plt.plot(train_loss_list, label='train_loss')
    plt.legend()
    plt.title('training loss and val loss')
    plt.savefig(os.path.join(out_path+args.model_name, 'loss.png'))
    plt.cla()
    plt.plot(val_acc_list, label='val_acc')
    plt.plot(train_acc_list, label='train_acc')
    plt.legend()
    plt.title('training acc and val acc')
    plt.savefig(os.path.join(out_path+args.model_name, 'acc.png'))
    
    torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(out_path+args.model_name, args.model_name+'.pth')
            )   
    print('Recall@1, 2, 4, 8: {recall[0]:.3f}, {recall[1]:.3f}, {recall[2]:.3f}, {recall[3]:.3f}; NMI: {nmi:.3f} \n'
                      .format(recall=recall, nmi=nmi))

def train(train_loader,correct,total,running_loss, model, criterion, optimizer, args):
    # switch to train mode
    model.train()
    if args.freeze_BN:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    losses = []
    for (img1, img2), y, (class1, class2) in train_loader:
        img1, img2, y = map(lambda x: x.to(args.gpu), [img1, img2, y])

        target = []
        IDimg_num = 3
        for i in range (int(len(y)/IDimg_num)):
            yy = y[(IDimg_num*i)]

            if i >0:
                target = torch.cat((target,yy),0)
            else:
                target = yy
        target = target.to(args.gpu)
        target = target.to(torch.long)

        output = model(img1, img2, batch_size = args.batch_size)
        loss, preds = criterion(output, target)
    
#         import pdb; pdb.set_trace()
        correct += (preds == target).sum().item()
        total += len(target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         running_loss += loss.item()
        losses.append(loss.item())
    
    train_loss = sum(losses)/max(1, len(losses))
    acc = correct / total
#     train_epoch_loss = running_loss / len(train_loader)
    
    print('train_acc : ',acc ,'  train_loss : ',train_loss)
    return acc , train_loss    
    


def validate(test_loader,correct,total,running_loss, model,criterion, args):
    # switch to evaluation mode
    model.eval()
    testdata = torch.Tensor()
    testlabel = torch.LongTensor()
    losses = []
    with torch.no_grad():
        for (img1, img2), y, (class1, class2) in test_loader:
            img1, img2, y = map(lambda x: x.to(args.gpu), [img1, img2, y])

            target = []
            IDimg_num = 3
            for i in range (int(len(y)/IDimg_num)):
                yy = y[(IDimg_num*i)]

                if i >0:
                    target = torch.cat((target,yy),0)
                else:
                    target = yy
            target = target.to(args.gpu)
            target = target.to(torch.long)

            output = model(img1, img2, batch_size = 6)
            loss, preds = criterion(output, target)
            correct += (preds == target).sum().item()
            total += len(target)
#             running_loss += loss.item()
            losses.append(loss.item())
    
            testdata = torch.cat((testdata, output.cpu()), 0)
            testlabel = torch.cat((testlabel, target.cpu()))
    
    val_loss = sum(losses)/max(1, len(losses))
#     val_epoch_loss = running_loss / len(test_loader)
    val_acc = correct / total
    print('val_acc : ',val_acc, 'val_loss : ',val_loss)
    nmi, recall = eva.evaluation(testdata.numpy(), testlabel.numpy(), [1, 2, 4, 8])
    return nmi, recall , val_acc, val_loss


def adjust_learning_rate(optimizer, epoch, args):
    # decayed lr by 10 every 20 epochs
    if (epoch+1)%20 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.rate
            
def dataload(train_dir,testdir):
    
    data_transform = transforms.Compose([
            transforms.RandomSizedCrop(480),
            transforms.RandomHorizontalFlip(),

    #         rotation_transform,
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(30),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
        ])


    valdata_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir,transform=data_transform)
    val_dataset = datasets.ImageFolder(root=testdir,transform=valdata_transforms)
#     train_set_size = int(len(train_dataset) * 0.8)
#     valid_set_size = len(train_dataset) - train_set_size
#     train_set, valid_set = data.random_split(train_dataset, [train_set_size, valid_set_size])
#     print('Train data set:', len(train_set))
#     print('val data set:', len(valid_set))
    return (train_dataset,val_dataset)

# def accuracy(pred, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         batch_size = target.size(0)

#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1))

#         res = []
#         correct_k = correct[:1].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size).item())
#         return res

if __name__ == '__main__':
    main()
