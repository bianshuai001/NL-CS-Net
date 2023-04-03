import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912#88912   # number of training blocks
batch_size = 64


## Load CS Sampling Matrix: phi                                                      
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
#Phi_data_Name = 'phi_0_10_1089' 
#Phi_data = sio.loadmat(Phi_data_Name)
#Phi_input=Phi_data['A']

Training_data_Name = 'Training_Data_Img91'
Training_data = sio.loadmat( Training_data_Name)
Training_labels = Training_data['labels']
#Phi_data_Name = 'down scale mat.mat' 
#Training_data = sio.loadmat(Phi_data_Name)
#Training_labels=Training_data['A']
#Training_data_Name = 'ista ++train400'
#Training_data = sio.loadmat(Training_data_Name)
#Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
#Training_labels = Training_data['d']

Qinit_Name = './%s/Initialization_Matrix_%d.mat' % (args.matrix_dir, cs_ratio)
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)
class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 100))
    return lr
def tensorlike(a):
    return a.size()[1]*a.size()[2]*a.size()[3]
def tv(x,flag):
  wx=torch.zeros_like(x)
  wy=torch.zeros_like(x)
  
  h_x=x.size()[2]
  w_x=x.size()[3]
  if flag==1:
   
   wx[:,:,0:h_x-1,:]=x[:,:,1:,:]-x[:,:,:h_x-1,:]
   wy[:,:,:,0:h_x-1]=x[:,:,:,1:]-x[:,:,:,:w_x-1]
   #wx[:,:,-1,:]=x[:,:,h_x-1,:]-x[:,:,0,:]
   #wy[:,:,:,-1]=x[:,:,:,w_x-1]-x[:,:,:,0]
   return wx,wy
  if flag==2:
   wx[:,:,1:,:]=x[:,:,:h_x-1,:]-x[:,:,1:,:]
   wy[:,:,:,1:]=x[:,:,:,:w_x-1]-x[:,:,:,1:]
   return wx,wy
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings

# Computing Initialization Matrix:
if os.path.exists(Qinit_Name):
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']

else:
    X_data = Training_labels.transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})

# Define ISTA-Net Block
class ResidualBlock_basic(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_basic, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        #cond = x[1]
        content = x

        out = self.act(self.conv1(content))
        out = self.conv2(out)
        return content + out
# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_scale=10,shape=64 ,p_len=64,in_channels=64
                 , inter_channels=16,use_multiple_size=False,use_topk=False,add_SE=False):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.lambda_step2 = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.β=nn.Parameter(torch.Tensor([0.01]))
        self.μ=nn.Parameter(torch.Tensor([0.01]))
        self.θ=nn.Parameter(torch.Tensor([0.01]))
        self.α=nn.Parameter(torch.Tensor([0.01]))
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 3, 3)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 3, 3)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 3, 3)))
        self.conv4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.conv5 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.conv6 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.nolocal1=nn.Parameter(init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
        self.nolocal2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 1, 3, 3)))
        self.nolocal3 = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.nolocal4 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        #self.conv1 = nn.Conv2d(64, 1, kernel_size=3)
        #self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        #self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(1)
        self.bn1=nn.BatchNorm2d(1,affine=True)#bn+conv_g+bn+relu+nlm+bn+conv+bn+relu+conv
        self.conv_G1 = nn.Parameter(init.xavier_normal_(torch.Tensor(64, 1, 3, 3)))
        self.bn2=nn.BatchNorm2d(64,affine=True)
        self.bn3=nn.BatchNorm2d(64,affine=True)
        self.bn4=nn.BatchNorm2d(1,affine=True)
        self.bn5=nn.BatchNorm2d(64,affine=True)
        self.conv_G2 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 64, 3, 3)))
        self.head_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock_basic(nf=32),
            ResidualBlock_basic(nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

        #nonlocal attribution
        self.ksize = ksize
        self.shape=shape
        self.p_len=p_len
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_scale = softmax_scale
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.use_multiple_size=use_multiple_size
        self.use_topk=use_topk
        self.add_SE=add_SE        
        self.conv33=nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)

                

    def forward(self, x,  PhiTb,v,r,ru,xx,uu,Phix,i, PhiWeight, PhiTWeight,):
        #x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        #x = x + self.lambda_step * PhiTb
        xorin=1*x
        x_input = x.view(-1, 1, 33, 33)
        #[dux,duy]=tv(x_input,1)
        du=F.conv2d(x_input, self.conv1, padding=1)
        #du=F.conv2d(du, self.conv4, padding=1)
        du=self.relu(du)
        #du=dux+duy
        v1=v.reshape(-1,1,33,33)
        duv=du-v1/self.β
        duv=duv.view(-1,1,33,33)
        duv = duv - self.lambda_step2 * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        duv = duv + self.lambda_step2 * PhiTb
        
        #x_D = F.conv2d(duv, self.conv_D, padding=1)
        x_mid = self.head_conv(duv)
        
        x_mid = self.ResidualBlocks(x_mid)
        x_mid = self.tail_conv(x_mid)

        x_pred = duv + x_mid
        #x = F.conv2d(x_D, self.conv1_forward, padding=1)
        #x = F.relu(x)
        #x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        #c=torch.sign(duv)
        #q=F.relu(torch.abs(duv) - 1/self.β)
        #x = torch.mul(c,q)

        #x = F.conv2d(x, self.conv1_backward, padding=1)
        #x = F.relu(x)
        #x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        #x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        #x_pred1 = x_input + x_G
        x_pred1=x_pred
        #x_pred = x_pred1.view(-1, 1089)
        #du1=du.view(-1,1089)
        duvw=self.β*du-v-self.β*x_pred
        duvw=duvw.view(-1,1,33,33)
        #[dtux,dtuy]=tv(duvw,2)
        #duvww=dtux+dtuy
        duvww=F.conv2d(duvw, self.conv2, padding=1)
        #duvww=F.conv2d(duvww, self.conv5, padding=1)
        duvww=self.relu(duvww)
        #duvww=duvww.view(-1,1089)
        aubru=self.μ*( F.conv2d(xorin, PhiWeight, padding=0, stride=33, bias=None)-Phix)-ru
        aubru = F.conv2d(aubru, PhiTWeight, padding=0, bias=None)
        wa = torch.nn.PixelShuffle(33)(aubru)
        d=duvww-r+self.θ*(xorin-xx)+ wa
        uu=uu-self.lambda_step*d
        
        rr=uu-r/self.θ
        rr=rr.view(-1,1,33,33)
      
         
        bn1a=self.bn1(rr)
        bn1a=F.conv2d(bn1a,self.conv_G1,padding=1)
        rr3=self.relu(bn1a)




         
         #nonlocal block
        #batch_size = rr3.size(0)

        #g_x = self.g(rr3).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        #g_x = g_x.permute(0, 2, 1)

        #theta_x = self.theta(rr3).view(batch_size, self.inter_channels, -1)
        #theta_x = theta_x.permute(0, 2, 1)

        #phi_x = self.phi(rr3).view(batch_size, self.inter_channels, -1)
        
        #f = torch.matmul(theta_x, phi_x)

        

        #f_div_C = F.softmax(f, dim=-1)

        #y = torch.matmul(f_div_C, g_x)
        #y = y.permute(0, 2, 1).contiguous()
        #y = y.view(batch_size, self.inter_channels, *rr3.size()[2:])
        #W_y = self.W(y)
        kernel = self.ksize

        b1 = self.g(rr3)
        b2 = self.theta(rr3)
        b3 = self.phi(rr3)

        raw_int_bs = list(b1.size())  # b*c*h*w

        patch_28, paddings_28 = extract_image_patches(b1, ksizes=[self.ksize, self.ksize],
                                                      strides=[self.stride_1, self.stride_1],
                                                      rates=[1, 1],
                                                      padding='same')
        patch_28 = patch_28.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_28 = patch_28.permute(0, 4, 1, 2, 3)
        patch_28_group = torch.split(patch_28, 1, dim=0)

        patch_112, paddings_112 = extract_image_patches(b2, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112 = patch_112.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112 = patch_112.permute(0, 4, 1, 2, 3)
        patch_112_group = torch.split(patch_112, 1, dim=0)

        patch_112_2, paddings_112_2 = extract_image_patches(b3, ksizes=[self.ksize, self.ksize],
                                                        strides=[self.stride_2, self.stride_2],
                                                        rates=[1, 1],
                                                        padding='same')

        patch_112_2 = patch_112_2.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_112_2 = patch_112_2.permute(0, 4, 1, 2, 3)
        patch_112_group_2 = torch.split(patch_112_2, 1, dim=0)
        f_groups = torch.split(b3, 1, dim=0)
        y = []
        for xii,xi, wi,pi in zip(f_groups,patch_112_group_2, patch_28_group, patch_112_group):
            w,h = xii.shape[2], xii.shape[3]
            _, paddings = same_padding(xii, [self.ksize, self.ksize], [1, 1], [1, 1])
            # wi = wi[0]  # [L, C, k, k]
            c_s = pi.shape[2]
            k_s = wi[0].shape[2]
            wi = wi.view(wi.shape[0],wi.shape[1],-1)
            xi = xi.permute(0, 2, 3, 4, 1)
            xi = xi.view(xi.shape[0],-1,xi.shape[4])
            score_map = torch.matmul(wi,xi)
            score_map = score_map.view(score_map.shape[0],score_map.shape[1],w,h)
            b_s, l_s, h_s, w_s = score_map.shape

            yi = score_map.view(b_s, l_s, -1)
            yi = F.softmax(yi*self.softmax_scale, dim=2).view(l_s, -1)
            pi = pi.view(h_s * w_s, -1)
            yi = torch.mm(yi, pi)
            yi=yi.view(b_s, l_s, c_s, k_s, k_s)[0]
            zi = yi.view(1, l_s, -1).permute(0, 2, 1)
            zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            inp = torch.ones_like(zi)
            inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0], stride=self.stride_1)
            zi = zi / out_mask
            y.append(zi)
        y = torch.cat(y, dim=0)
        y = self.W(y)
        z = self.θ*rr + y
        z=F.conv2d(z,self.conv_G2,padding=1)


        #z = 2*self.α*W_y.view(batch_size,self.inter_channels,rr.size()[2],rr.size()[3]) + self.θ*rr
        xx2=z/(self.θ+2*self.α)
        xx2=self.bn4(xx2)
        #xx2=F.conv2d(xx2,self.nolocal1,padding=1)
        #xx2=self.bn5(xx2)
        xx2=self.relu(xx2)
        xx=F.conv2d(xx2,self.nolocal2,padding=1)
        
        #xx=xx.view(-1,1089)
        #xx1=xx.view(-1,1089)

       
        
         


        #update mutipl
        uu1=uu.view(-1,1,33,33)
        uu3=uu.view(-1,1,33,33)
        #[uu2,uu3]=tv(uu1,1)
        #uu1=uu2+uu3
        uu1=F.conv2d(uu1, self.conv3, padding=1)
        #uu1=F.conv2d(uu1, self.conv6, padding=1)
        uu1=self.relu(uu1)
        #uu1=uu1.view(-1,1089)
        v=v-self.β*(uu1-x_pred)
        r=r-self.θ*(uu-xx)
        ru=ru-self.μ*( F.conv2d(uu3, PhiWeight, padding=0, stride=33, bias=None)-Phix)
        #x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        #x = F.relu(x)
        #x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        #symloss = x_D_est - x_D
        #[uu, symloss]
        return [uu,PhiTb,v,r,ru,xx,uu,Phix,i, PhiWeight, PhiTWeight]


# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, 1089)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))
        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, batch_x):
        x=batch_x
        x=x.view(-1,1,33,33)
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        #PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=33, bias=None)
        #PhiTb = torch.mm(Phix,Phi)
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb
        #Phix1=Phix.view(batch_size,1,16,16)
        #x = torch.mm(Phix,torch.transpose(Qinit, 0, 1))
        #x= F.interpolate(Phix1, scale_factor=2)
        #x=x.view(batch_size,-1)
        v=torch.zeros_like(batch_x)
        v=v.view(-1,1,33,33)
        r=torch.zeros_like(batch_x)
        r=r.view(-1,1,33,33)
        col=batch_x.size()[0]
        rol=Phi.size()[0]
        ru=torch.zeros_like(Phix)
        
        xx=torch.zeros_like(batch_x)
        xx=xx.view(-1,1,33,33)
        uu=torch.zeros_like(batch_x)
        uu=uu.view(-1,1,33,33)
        #layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [uu,  PhiTb,v,r,ru,xx,uu,Phix,i, PhiWeight, PhiTWeight,] = self.fcs[i](x,  PhiTb,v,r,ru,xx,uu,Phix,i, PhiWeight, PhiTWeight,)
            #layers_sym.append(layer_sym)

        x_final = uu


        return [x_final, Phi]#, layers_sym


model = ISTANetplus(layer_num, n_input)
model = nn.DataParallel(model)
model = model.to(device)



print_flag = 1   # print parameter number

if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/nonlocal cola op net%d_group_%d_ratio_%d_lr_%.4f" % (args.model_dir, layer_num, group_num, cs_ratio, learning_rate)

log_file_name = "./%s/Log_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, cs_ratio, learning_rate)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.cuda()

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.cuda()
Eye_I = torch.eye(n_input).to(device)

# Training loop
for epoch_i in range(start_epoch+1, end_epoch+1):
    for data in rand_loader:
        lr = adjust_learning_rate(optimizer, epoch_i)

        for param_group in optimizer.param_groups:
          param_group["lr"] = lr
        batch_x = data
        batch_x = batch_x.cuda()
        
        #Phix = torch.mm(batch_x,torch.transpose(Phi,0,1))
        #with autocast():
        [x_output,Phi] = model(batch_x)
        x_output=x_output.view(-1,1089)
        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
        loss_orth = torch.mean(torch.pow(torch.mm(Phi, torch.transpose(Phi, 0, 1))-Eye_I, 2))
        mu = torch.Tensor([0.01]).to(device)
        #loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        #for k in range(layer_num-1):
            #loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        #gamma = torch.Tensor([0.01]).to(device)

        #loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(mu, loss_orth)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        #scaler.scale(loss_all).backward()
        optimizer.step()
        #scaler.step(optimizer)
        #scaler.update()
        output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
        #output_data = "[%02d/%02d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint)
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters