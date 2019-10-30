import torch.nn as nn
import torch

""" D3 and Densenet Models """
class denseNet(nn.Module):
  '''
  https://arxiv.org/pdf/1608.06993.pdf
  
  '''
  
  def __init__(self, in_c, L=5, K=12):
    '''
    denseNet block
    
    Params:
      in_c : no. of input chanels
      L : no. of denseNet layers
      K : no. of output chanels
    '''
    super(denseNet, self).__init__()
    self.l = L
    self.k = K
    
    self.L1_conv1 = nn.Conv2d(in_c+2, self.k*4, kernel_size=1, stride=1)
    self.L1_conv2 = nn.Conv2d(self.k*4, self.k, kernel_size=3, stride=1, padding=1) 
    
    self.L2_conv1 = nn.Conv2d(self.k+2, self.k*4, kernel_size=1, stride=1)
    self.L2_conv2 = nn.Conv2d(self.k*4, self.k, kernel_size=3, stride=1, padding=1)
    
    self.L3_conv1 = nn.Conv2d(self.k+2, self.k*4, kernel_size=1, stride=1)
    self.L3_conv2 = nn.Conv2d(self.k*4, self.k, kernel_size=3, stride=1, padding=1)
    
    self.L4_conv1 = nn.Conv2d(self.k+2, self.k*4, kernel_size=1, stride=1)
    self.L4_conv2 = nn.Conv2d(self.k*4, self.k, kernel_size=3, stride=1, padding=1)
    
    self.L5_conv1 = nn.Conv2d(self.k+2, self.k*4, kernel_size=1, stride=1)
    self.L5_conv2 = nn.Conv2d(self.k*4, self.k, kernel_size=3, stride=1, padding=1)
   
  
  def forward(self, x, sp_inputs):
    
    sp_inputs = torch.tensor(sp_inputs).float()
    
    x1 = torch.cat((x, sp_inputs), 1)
    x1 = torch.relu(self.L1_conv1(x1))
    x1 = torch.relu(self.L1_conv2(x1))
    
    x2 = torch.cat((x1, sp_inputs), 1)
    x2 = torch.relu(self.L2_conv1(x2))
    x2 = torch.relu(self.L2_conv2(x2))
    
    x3 = torch.cat((x2, sp_inputs), 1)
    x3 = torch.relu(self.L3_conv1(x3))
    x3 = torch.relu(self.L3_conv2(x3))
    
    x4 = torch.cat((x3, sp_inputs), 1)
    x4 = torch.relu(self.L4_conv1(x4))
    x4 = torch.relu(self.L4_conv2(x4))
    
    x5 = torch.cat((x4, sp_inputs), 1)
    x5 = torch.relu(self.L5_conv1(x5))
    x5 = torch.relu(self.L5_conv2(x5))
    
    return x5

class D3(nn.Module):
  """
  https://arxiv.org/pdf/1804.02771v2.pdf

  """
  
  def __init__(self):
    """
    D3 block
  
    """
    
    super(D3, self).__init__()
    
    self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1)
    
    # Downsampling denseNet block 1
    self.dense1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet1 = denseNet(64)
    self.dense1_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1)
    
    # Downsampling denseNet block 2
    self.dense2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet2 = denseNet(64)
    self.dense2_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1)
    
    # Skip dense2 block
    self.skp_d2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.skp_densenet2 = denseNet(64)
    self.skp_d2_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)

    # Downsampling denseNet block 3
    self.dense3_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet3 = denseNet(64)
    self.dense3_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1)
    
    # Skip dense3 block
    self.skp_d3_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.skp_densenet3 = denseNet(64)
    self.skp_d3_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
    
    # Downsampling denseNet block 4
    self.dense4_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet4 = denseNet(64)
    self.dense4_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=2, padding=1)
    
    # Skip dense4 block
    self.skp_d4_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.skp_densenet4 = denseNet(64)
    self.skp_d4_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
    
    # Downsampling denseNet block 5
    self.dense5_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet5 = denseNet(64)
    self.dense5_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
    
    # Downsampling denseNet block 6
    self.dense6_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.densenet6 = denseNet(64)
    self.dense6_conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1)
    
    # Upsampling denseNet block 1
    self.up_dense1_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.up_densenet1 = denseNet(64)
    self.up_dense1_conv2 = nn.ConvTranspose2d(12, 64, kernel_size=2, stride=2, padding=0)
    
    # Upsampling denseNet block 2
    self.up_dense2_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.up_densenet2 = denseNet(64)
    self.up_dense2_conv2 = nn.ConvTranspose2d(12, 64, kernel_size=2, stride=2, padding=0)
    
    # Upsampling denseNet block 3
    self.up_dense3_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.up_densenet3 = denseNet(64)
    self.up_dense3_conv2 = nn.ConvTranspose2d(12, 64, kernel_size=2, stride=2, padding=0)
    
    # Upsampling denseNet block 4
    self.up_dense4_conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.up_densenet4 = denseNet(64)
    self.up_dense4_conv2 = nn.ConvTranspose2d(12, 64, kernel_size=2, stride=2, padding=0)
    
    # Transpose convolution 
    self.up_conv1 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2, padding=0)
    
  def forward(self, x, sp_inputs):
    
    sp_inputs = sp_inputs.float()
    
    # conv1 
    x = torch.cat((x, sp_inputs), 1)
    x = torch.relu(self.conv1(x))
    
    # densenet1
    h =  nn.functional.interpolate(sp_inputs, scale_factor=0.5,mode="bilinear",align_corners=True)
    x1 = torch.relu(self.dense1_conv1(x))
    x1 = self.densenet1(x1, h)
    x1 = torch.relu(self.dense1_conv2(x1))
    
    # skip densenet2
    h2 = nn.functional.interpolate(h, scale_factor=0.5,mode="bilinear",align_corners=True)
    skip_x2 = torch.relu(self.skp_d2_conv1(x1))
    skip_x2 = self.skp_densenet2(skip_x2, h2)
    skip_x2 = torch.relu(self.skp_d2_conv2(skip_x2))
    
    # densenet2
    x2 = torch.relu(self.dense2_conv1(x1))
    x2 = self.densenet2(x2, h2)
    x2 = torch.relu(self.dense2_conv2(x2))
    
    # skip densenet3
    h3 = nn.functional.interpolate(h2, scale_factor=0.5,mode="bilinear",align_corners=True)
    skip_x3 = torch.relu(self.skp_d3_conv1(x2))
    skip_x3 = self.skp_densenet3(skip_x3, h3)
    skip_x3 = torch.relu(self.skp_d3_conv2(skip_x3))
    
    # densenet3
    x3 = torch.relu(self.dense3_conv1(x2))
    x3 = self.densenet3(x3, h3)
    x3 = torch.relu(self.dense3_conv2(x3))
    
    # skip densenet4
    h4 = nn.functional.interpolate(h3, scale_factor=0.5,mode="bilinear",align_corners=True)
    skip_x4 = torch.relu(self.skp_d4_conv1(x3))
    skip_x4 = self.skp_densenet4(skip_x4, h4)
    skip_x4 = torch.relu(self.skp_d4_conv2(skip_x4))
    
    # densenet4
    x4 = torch.relu(self.dense4_conv1(x3))
    x4 = self.densenet4(x4, h4)
    x4 = torch.relu(self.dense4_conv2(x4))
    
    # densenet5
    h5 = nn.functional.interpolate(h4, scale_factor=0.5,mode="bilinear",align_corners=True)
    x5 = torch.relu(self.dense5_conv1(x4))
    x5 = self.densenet5(x5, h5)
    x5 = torch.relu(self.dense5_conv2(x5))
   
    # densenet6
    x6 = torch.relu(self.dense6_conv1(x5))
    x6 = self.densenet6(x6, h5)
    x6 = torch.relu(self.dense6_conv2(x6))
    
    # up dense block 1
    x_u1 = torch.relu(self.up_dense1_conv1(x6))
    x_u1 = self.up_densenet1(x_u1, h5)
    x_u1 = torch.relu(self.up_dense1_conv2(x_u1))
    
    x_u1 = torch.cat((x_u1, skip_x4), 1)
    
    # up dense block 2
    x_u2 = torch.relu(self.up_dense2_conv1(x_u1))
    x_u2 = self.up_densenet2(x_u2, h4)
    x_u2 = torch.relu(self.up_dense2_conv2(x_u2))

    x_u2 = torch.cat((x_u2, skip_x3), 1)
    
    # up dense block 3
    x_u3 = torch.relu(self.up_dense3_conv1(x_u2))
    x_u3 = self.up_densenet3(x_u3, h3)
    x_u3 = torch.relu(self.up_dense3_conv2(x_u3))
 
    x_u3 = torch.cat((x_u3, skip_x2), 1)
    
    # up dense block 4
    x_u4 = torch.relu(self.up_dense4_conv1(x_u3))
    x_u4 = self.up_densenet4(x_u4, h2)
    x_u4 = torch.relu(self.up_dense4_conv2(x_u4))
    
    x_f = torch.relu(self.up_conv1(x_u4))
    
    return x_f
