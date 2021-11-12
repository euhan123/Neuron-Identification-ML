import os, time, pickle, argparse, networks, utils
import glob
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from image_folder import Image_Folder
from torch.utils.data import DataLoader
import cv2
from utils import save_checkpoint as sc

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='euhanNet_3x3_scale_down',  help='')
parser.add_argument('--train_path', required=False, default='./dataset/train/')
parser.add_argument('--test_path', required=False, default='./dataset/test/')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--train_epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
args = parser.parse_args()

def main():
    device = torch.device('cpu')
    if torch.backends.cudnn.enabled:
       torch.backends.cudnn.benchmark = True

    # results save path
    if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
        os.makedirs(os.path.join(args.name + '_results', 'Transfer'))


    # data_loader
    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    train_dataset_src = Image_Folder('train',args.train_path, transform)
    train_loader_src = DataLoader(dataset=train_dataset_src, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset_src = Image_Folder('train', args.test_path, transform)
    test_loader_src = DataLoader(dataset=test_dataset_src, batch_size=1, shuffle=False, drop_last=True)

    # network
    model = networks.euhanNet()
    model.to(device)

    print('---------- Networks initialized -------------')
    utils.print_network(model)
    print('-----------------------------------------------')

    # loss
    weights = [1.0, 9.0]
    class_weights = torch.FloatTensor(weights).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255).to(device)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay = 0.0005)

    train_hist = {}
    train_hist['loss'] = []
    train_hist['per_epoch_time'] = []
    print('training start!')
    start_time = time.time()

    epochs = []
    loss_list = []

    for epoch in range(args.train_epoch):
        epoch_start_time = time.time()
        model.train()

        losses = []

        losses = train(train_loader_src, model, optimizer, device, losses, train_hist, ce_loss)
        accuracy = validate(test_loader_src, model, device, epoch)

        per_epoch_time = time.time() - epoch_start_time
        train_hist['per_epoch_time'].append(per_epoch_time)

        epochs.append(epoch)
        loss_list.append(torch.mean(torch.FloatTensor(losses)))

        plt.figure()
        plt.plot(epochs, loss_list)
        plt.savefig('./loss_graph.png')
        plt.close()

        print(
            '[%d/%d] - time: %.2f, loss: %.3f, accuracy: %.3f' % (
            (epoch), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(losses)), torch.mean(torch.FloatTensor(accuracy))))

        sc({
            'state_dict': model.state_dict(),
        }, 0, filename='./checkpoint/checkpoint' + str(epoch) + '.pth.tar')



def train(train_loader_src, model, optimizer, device, losses, train_hist, ce_loss):
    for data in train_loader_src:
        img3x3, img1x1, gt, _ = data
        img3x3, img1x1, gt = img3x3.to(device), img1x1.to(device), gt.to(device)

        # train
        optimizer.zero_grad()

        output = model(img3x3, img1x1)
        loss = ce_loss(output, gt.long())

        losses.append(loss.item())
        train_hist['loss'].append(loss.item())

        loss.backward()
        optimizer.step()
    return losses

def validate(test_loader_src, model, device, epoch):
    model.eval()
    accuracy_ = []
    for data in test_loader_src:
        img3x3, img1x1, gt, name = data
        img3x3, img1x1 = img3x3.to(device),img1x1.to(device)

        output = model(img3x3, img1x1)
        output = output.data.cpu().numpy()
        gt = gt.data.cpu().numpy()
        gt = np.squeeze(gt)
        output = np.argmax(output, axis = 1)
        output = np.squeeze(output)

        size = output.shape[0]

        accuracy = 100 - ((sum(sum(abs(output-gt))))/(size**2))*100
        accuracy_.append(accuracy)

        output = np.expand_dims(output, axis = 2)
        gt = np.expand_dims(gt, axis = 2)
        output = np.concatenate((output, output, output), axis = 2)
        gt = np.concatenate((gt,gt,gt), axis = 2)
        output = np.concatenate((output, gt), axis = 1)
        output *= 255

        name = name[0]
        name = str(epoch) + "_" + name
        cv2.imwrite(os.path.join(args.name + '_results', 'Transfer', name), output)

    return accuracy_







if __name__ == '__main__':
    main()