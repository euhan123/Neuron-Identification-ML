import os, argparse, networks, utils
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='NonNon',  help='')
parser.add_argument('--src_data', required=False, default='./src_data',  help='sec data path')
parser.add_argument('--tgt_data', required=False, default='./tgt_data',  help='tgt data path')
parser.add_argument('--vgg_model', required=False, default='vgg19-dcbb9e9d.pth', help='pre-trained VGG19 model path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--train_epoch', type=int, default=100)
parser.add_argument('--pre_train_epoch', type=int, default=10)
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--con_lambda', type=float, default=10, help='lambda for content loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
# parser.add_argument('--pre_trained_model', required=True, default='NonNon_results', help='pre_trained cartoongan model path')
parser.add_argument('--image_dir', required=False, default='./data/src_data/', help='test image path')
parser.add_argument('--output_image_dir', required=False, default='./data/output/all_test', help='output test image path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True

for k in [5]:
    pre_trained_model = './CA_Gen+seg_Dis_results/'+str(k)+'_generator.pkl'
    G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(pre_trained_model))
    else:
        # cpu mode
        G.load_state_dict(torch.load(pre_trained_model, map_location=lambda storage, loc: storage))
    G.to(device)

    src_transform = transforms.Compose([
            transforms.Resize((args.input_size*2, args.input_size*2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
    image_src = utils.data_load(os.path.join(args.image_dir), 'test_tree', src_transform, 1, shuffle=False, drop_last=True)

    with torch.no_grad():
        G.eval()
        for n, (x, _) in enumerate(image_src):
            x = x.to(device)
            G_recon = G(x)
            # result = torch.cat((x[0], G_recon[0]), 2)
            result = G_recon[0]
            path = os.path.join(args.output_image_dir, str(n + 1) +'_' +str(k) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)


