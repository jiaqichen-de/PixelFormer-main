import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer 
from .swin_transformer import SwinTransformerBlock
from .PQI import PSP
from .SAM import SAM
from einops import rearrange
import numpy as np
from .attractor import AttractorLayer
from .dist_layers import ConditionalLogBinomial
#This is the implementation of model PixelFormerPlus
#The optimiztion of model PixelFormer
#https://arxiv.org/pdf/2210.09071.pdf
"""Attention Attention Everywhere:
Monocular Depth Prediction with Skip Attention"""
#https://github.com/ashutosh1807/PixelFormer

class BCP(nn.Module):
    """ Bin Center Prediction Module
        there are there modes Multihead Pixelbin others by setting the value of bcpmode
        Inspired by Multihead Mechanism, Multihead means we need to divide the feature into
        several groups, and generate bin divisions and their 
        corresponding predicted probabilities for each groups.
        Pixelbin means the bin center prediction module assigns a bin partition for each pixel
        Others means the all pixels share the common bin partition
    """

    def __init__(self, max_depth, min_depth, bcpmode,split,in_features=512, hidden_features=512*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.bcpmode=bcpmode
        self.split=split
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc11=nn.Linear(512//len(self.split), 2048//len(self.split))
        if self.bcpmode=="MultiHead":
          #self.fc21=nn.Linear(2048//len(self.split), 1024//len(self.split))
          #self.fc21=nn.Linear(2048//len(self.split), 512//len(self.split))
          self.fc21=nn.Linear(2048//len(self.split), 256//len(self.split))
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pixelbin=nn.Sequential(
            nn.ConvTranspose2d(in_features, 256,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.ConvTranspose2d(256, 256,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.ConvTranspose2d(256, 128,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)
           # nn.Dropout(0.1),)
        """self.pixelbin=nn.Sequential(
            nn.Conv2d(512, 128, 3,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),)"""
        """if we employ transposed convolution to build the bin center prediction module,
           and keep the first self.pixelbin and comment the other self.pixelbin out. 
           if we use bilinear interpolation for bin center prediction,
           and keep the second self.pixelbin and add bilinear interpolation on line 103.
           When we wanna have the model structure on which
           the feature map that records the bin center partitions for each pixel 
           needs to be upsampled in bin adjustment module, we should keep 
           the second self.pixelbin and comment the other self.pixelbin out"""

    def forward(self, x):
        if self.bcpmode== "MultiHead":

          x=split(x,self.split,"MultiHead")
          bincenters=[]
          
          for i in range(len(self.split)):
             x[i]= torch.mean(x[i].flatten(start_dim=2), dim = 2)
             
             x[i]= self.fc11(x[i])
             
             x[i]= self.act(x[i])
             x[i]= self.drop(x[i])
             x[i] =self.fc21(x[i])
             
             x[i]= self.drop(x[i])
             #eps = 1e-3
             #bins = x[i] + eps
             bins = torch.softmax(x[i], dim=1)
             bins = bins / bins.sum(dim=1, keepdim=True)
             bin_widths = (self.max_depth - self.min_depth) * bins
             bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
             bin_edges = torch.cumsum(bin_widths, dim=1)
             centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
             n, dout = centers.size()
             centers = centers.contiguous().view(n, dout, 1, 1)
            
             bincenters.append(centers)
          return bincenters
        elif self.bcpmode== "Pixelbin":
           #x=F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)
           x=self.pixelbin(x)
           eps = 1e-3
           bins = x + eps
           bins = bins / bins.sum(dim=1, keepdim=True)
           bin_widths = (self.max_depth - self.min_depth) * bins
           bin_widths = nn.functional.pad(bin_widths, (0,0,0,0,1,0), mode='constant', value=self.min_depth)
           bin_edges = torch.cumsum(bin_widths, dim=1)
           centers = 0.5 * (bin_edges[:, :-1, ...] + bin_edges[:,1:,...])
           
           return centers
           
        else:
          
          x = torch.mean(x.flatten(start_dim=2), dim = 2)
          
          x = self.fc1(x)
          x = self.act(x)
          x = self.drop(x)
          x = self.fc2(x)
          x = self.drop(x)
          bins = torch.softmax(x, dim=1)
          bins = bins / bins.sum(dim=1, keepdim=True)
          bin_widths = (self.max_depth - self.min_depth) * bins         
          bin_widths = nn.functional.pad(bin_widths, (1,0), mode='constant', value=self.min_depth)          
          bin_edges = torch.cumsum(bin_widths, dim=1)         
          centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
          
          n, dout = centers.size()
          centers = centers.contiguous().view(n, dout, 1, 1)
          return centers

class PixelFormerPlus(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        
        EncoderGlobal_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = EncoderGlobal_cfg['num_classes']*4
        win = 7
        sam_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        self.sam4 = SAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.sam3 = SAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.sam2 = SAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.sam1 = SAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        # There are four upsampling modes
        # Transposed Concolution
        # Bilinear Interpolation
        #PixelShuffle 
        #Patch expanding
        self.upsample1=DecodingPhaseUpsample(upsample_mode="Transpose Convolution",input_channels=1024,output_channels=256)
        self.upsample2=DecodingPhaseUpsample(upsample_mode="Transpose Convolution",input_channels=512,output_channels=128)
        self.upsample3=DecodingPhaseUpsample(upsample_mode="Transpose Convolution",input_channels=256,output_channels=64)

        self.EncoderGlobal = PSP(**EncoderGlobal_cfg)

        bcp_mode= "Pixelbin"
        
        if bcp_mode== "Pixelbin":
           self.bcp_mode="BinAdjustment"
        else:
           self.bcp_mode="None"
        """Multihead means we need to divide the feature into
        several groups, and generate bin divisions 
        and their corresponding predicted probabilities for each groups"""
        # we split the feature map to several groups, we should decide how to split
        # The number of list elements represents the number of heads
        # Each item represents the number of bins
        if bcp_mode== "MultiHead":
          #split_numer= [256,256,256,256]
          #split_numer= [128,128,128,128]
          split_numer= [64,64,64,64]
          #split_numer= [256,256,256,256,256,256,256,256]
          #split_numer= [128,128,128,128,128,128,128,128]
          #split_numer= [64,64,64,64,64,64,64,64]
          #If we implementate the nested decoder subnetworks, we set Pyramid Structure
        else:
          split_numer= [64,64,64,64]
           
        self.DecodingPhaseOutputMode = "Pyramid Structure"
        
        self.disp_head1 = DispHead(input_dim=128,bcpmode=bcp_mode,split=split_numer)
        self.BinAdjustment=BinAdjustment(input_dim=[1024, 512, 256, 128], n_attractors=[16, 8, 4, 1])
        # There are four upsampling modes for nested decoder subnetworks
        # Transposed Concolution
        # Bilinear Interpolation
        # PixelShuffle 
        # Patch expanding
        self.Outputupsample1=DecodingPhaseUpsample(upsample_mode="Bilinear Interpolation",input_channels=1024,output_channels=512)
        self.Outputupsample2=DecodingPhaseUpsample(upsample_mode="Bilinear Interpolation",input_channels=512,output_channels=256)
        self.Outputupsample3=DecodingPhaseUpsample(upsample_mode="Bilinear Interpolation",input_channels=256,output_channels=128)   
        self.bcp = BCP(max_depth=max_depth, min_depth=min_depth,bcpmode=bcp_mode,split=split_numer)

        self.init_weights(pretrained=pretrained)
        self.change_channels=nn.Sequential(
            nn.Conv2d(512,1024,3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.EncoderGlobal.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def forward(self, imgs):

        enc_feats,xoutputs= self.backbone(imgs)
        

        q4 = self.EncoderGlobal(enc_feats)
            
        if self.DecodingPhaseOutputMode == "Pyramid Structure":
          #The nested Decoder Subnetworks

          q3 = self.sam4(enc_feats[3], q4)
            
          """these output features with further refined features from the subsequent stage 
          linearly combines with the features from the skip attention mechanism module"""
          f = self.change_channels(q4)+q3
          x_blocks=[]
          x_blocks.append(q3) 
          """When the bin adjustment module requires refined features from the decoder stage, 
          should we input features from the SAM module (q) or 
          from the nested decoder sub-network (f)? 
          """
          #x_blocks.append(f)
          # Then upsamples the combined features 
          f = self.Outputupsample1(f)
          
          q3 = self.upsample1(q3)
          q2 = self.sam3(enc_feats[2], q3)
          #Linear combination
          f = f+q2
          x_blocks.append(q2) 
          #x_blocks.append(f)
          # Then upsamples the combined features
          f = self.Outputupsample2(f)
          
          q2 = self.upsample2(q2)
          q1 = self.sam2(enc_feats[1], q2)
          #Linear combination
          f = f+q1
          x_blocks.append(q1) 
          #x_blocks.append(f)
          # Then upsamples the combined features
          f = self.Outputupsample3(f)
          
          q1 = self.upsample3(q1)
          q0 = self.sam1(enc_feats[0], q1)
          _, _, H, W = q0.size()
          
        #Linear combination
          f = f+q0
          x_blocks.append(q0) 
          #x_blocks.append(f)
          bin_centers = self.bcp(q4)
          # There are two modes
          # BinAdjustment means refine the generated bins with refined features in Decoder
          # Others means don't adjust the generated bins
          if self.bcp_mode=="BinAdjustment":
               f=self.BinAdjustment(f, bin_centers,x_blocks,H,W,q4)
          else:
               f = self.disp_head1(f, bin_centers)
          
          
          return f
         
        else: 
             
          #Without the nested Decoder Subnetworks
          q3 = self.sam4(enc_feats[3], q4)
          x_blocks=[]
          x_blocks.append(q3)
          q3 = self.upsample1(q3)
          q2 = self.sam3(enc_feats[2], q3)
          x_blocks.append(q2)
          q2 = self.upsample2(q2)
          q1 = self.sam2(enc_feats[1], q2)
          x_blocks.append(q1)
          q1 = self.upsample3(q1)
          q0 = self.sam1(enc_feats[0], q1)
          _, _, H, W = q0.size()
          x_blocks.append(q0)
          bin_centers = self.bcp(q4)
          # There are two modes
          # BinAdjustment means refine the generated bins with refined features in Decoder
          # Others means don't adjust the generated bins
          if self.bcp_mode=="BinAdjustment":
              f=self.BinAdjustment(q0, bin_centers,x_blocks,H,W,q4)
          else:
              f = self.disp_head1(q0, bin_centers)
          

          return f


          

class DecodingPhaseUpsample(nn.Module):
    """        
    Upsampling for SAM and the nested decoder Subnwtworks
    H*W*C -> 2*H*2*W*C/4 
    H*W*C -> 2*H*2*W*C/2
    """
    def __init__(self, upsample_mode,input_channels,output_channels):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.patch_expand = PatchExpand(dim=input_channels)
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.convtranspose=nn.Sequential(
            nn.ConvTranspose2d(self.input_channels, self.output_channels,kernel_size=2,stride=2,padding=0),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True),)
        self.conv=nn.Sequential(
            nn.Conv2d(self.input_channels, self.output_channels, 3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True),)
        
        self.conv1=nn.Sequential(
            nn.Conv2d(self.input_channels//4, self.output_channels, 3, padding=1),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True),)
        
    def forward(self,x):
        if self.upsample_mode == "Transpose Convolution": 
        
          
          return self.convtranspose(x)
        elif self.upsample_mode == "Bilinear Interpolation":
          
          x=F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
          
          return self.conv(x)
        elif self.upsample_mode == "PixelShuffle":
           if  self.input_channels//4 == self.output_channels:
            return nn.PixelShuffle(2)(x)
           else:
            return self.conv1(nn.PixelShuffle(2)(x))
        else:
           if  self.input_channels//4 == self.output_channels: 
            return self.patch_expand(x)
           else:
            return self.conv1(self.patch_expand(x))
          
          
#patch expanding 
#Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation

#This class is the implementation of Swin-Unet
#https://arxiv.org/abs/2105.05537v1


class PatchExpand(nn.Module):  
 
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        #self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        """
        x: B, H*W, C
        """
        
        #x = self.expand(x)
        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)
        x=x.view(-1, 2*H, 2*W, C//4).permute(0, 3, 1, 2).contiguous()
        return x
    


class DispHead(nn.Module):
    """Linear Combination of bin Centers and generated probability values"""
    def __init__(self, input_dim,bcpmode,split):
        super().__init__()
        self.input_dim = input_dim
        self.bcpmode=bcpmode
        self.split=split
        """self.conv=nn.Sequential(
            nn.Conv2d(128, 1024, 3,padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)"""
        #self.conv=nn.Conv2d(128, 2048, 3,padding=1, bias=False)
        #self.conv=nn.Conv2d(128, 1024, 3,padding=1, bias=False)
        #self.conv=nn.Conv2d(128, 512, 3,padding=1, bias=False)
        self.conv=nn.Conv2d(128, 256, 3,padding=1, bias=False)
        """self.conv1=nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)"""
        #self.conv1=nn.Conv2d(128, 512, 3,padding=1, bias=False)
        self.conv1=nn.Conv2d(128, 256, 3,padding=1, bias=False)
        #self.conv1=nn.Conv2d(128, 128, 3,padding=1, bias=False)
        self.upsample=Upsample(upsample_mode="bilinear",input_channels=1)
        self.conditional_log_binomial = ConditionalLogBinomial(
            128, 128, n_classes=128, min_temp=5, max_temp=50)
    def forward(self, x, centers):
        if self.bcpmode == "MultiHead":
           
           x = self.conv(x)
           
           x=split(x,self.split,"else")
           depths=[]
           
           for i in range(len(self.split)):
              #x[i] = self.conditional_log_binomial(x[i],q0)
              x[i]=torch.softmax(x[i],dim=1)
              input=x[i]
              center_depth=centers[i]
              
              sum=torch.sum(input*center_depth,dim=1,keepdim=True,dtype=float)
              
              depths.append(sum)
           depth=torch.cat(depths,dim=1)
           depth=torch.mean(depth,dim=1,keepdim=True)
           
           depth = self.upsample(depth)
           
           return depth
        else:
           
          
          x = self.conv1(x)
        
          x = x.softmax(dim=1)
          
          x = torch.sum(x * centers, dim=1, keepdim=True)
          x=self.upsample(x)
          
          
          return x



class BinAdjustment(nn.Module):
    """Bin Adjustment Module and Linear Combination of bin Centers and generated probability values
       Bin Adjustment Module refine the generated bins with refined features in Decoder
       """
    def __init__(self, input_dim, n_attractors):
        super().__init__()
        self.attractors = nn.ModuleList([
            AttractorLayer(in_features=input_dim[i], n_bins=128, n_attractors=n_attractors[i])
                      
            for i in range(len(n_attractors))
        ])
        input=[512, 1024, 512, 256]
        self._net =nn.ModuleList([
            nn.Conv2d(input[i], input_dim[i], 1, 1, 0)
                      
            for i in range(len(n_attractors))
        ]) 
        
        self.conv=nn.Conv2d(128, 256, 3,padding=1, bias=False)
        """nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),)"""
        self.upsample=Upsample(upsample_mode="bilinear",input_channels=1)
        self.conditional_log_binomial = ConditionalLogBinomial(
            128, 128, n_classes=128, min_temp=5, max_temp=50)
    def forward(self, x, centers, x_blocks,h,w,q4):
       #x = self.conv(x)
       prev_b_embedding=q4
       for  net,attractor, out in zip( self._net,self.attractors, x_blocks):
            
            prev_b_embedding=net(prev_b_embedding)
            """b= attractor(
                out,centers,h,w)"""
            b= attractor(
                out,centers,h,w,prev_b_embedding=prev_b_embedding, interpolate=True)
            centers = b.clone()
            prev_b_embedding = out.clone()

       x=self.upsample(x)
       prev_b_embedding =self.upsample(prev_b_embedding)
       x = self.conditional_log_binomial(x, prev_b_embedding)
       centers=self.upsample(centers)
       x = torch.sum(x * centers, dim=1, keepdim=True)
         
       return x

#Function for spliting the feature maps

def split(x,splitnumbers,splitmode):
   if splitmode=="MultiHead":
     splitx=[]
     a=0
     for i in range(len(splitnumbers)):
     
      xterm=x[:,a:a+128,:,:]
      a=a+128
      #xterm=x[:,a:a+64,:,:]
      #a=a+64
     
      splitx.append(xterm)
     return splitx
     
   else:
    splitx=[]
    a=0
    for i in range(len(splitnumbers)):
     
     xterm=x[:,a:a+splitnumbers[i],:,:]
     a=a+splitnumbers[i]
     
     splitx.append(xterm)
    return splitx
    
   
class Upsample(nn.Module):
    
    def __init__(self, upsample_mode,input_channels):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.input_channels=input_channels
        self.convtranspose=nn.ConvTranspose2d(self.input_channels, self.input_channels,kernel_size=4,stride=4,padding=0)
        
               
    def forward(self,x):
        if self.upsample_mode == "Transpose Convolution": 
                     
            return self.convtranspose(x)
         
        else:
            
            
            return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
            
            
