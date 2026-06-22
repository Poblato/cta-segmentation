import numpy
import torch
import torch.nn as nn

class convolution_block(nn.Module):
    """
    A basic convolutional block used in the U-Net architecture.
    Consists of two 3x3 convolutional layers, each followed by Batch Normalization.
    A ReLU activation function is applied at the end.
    """
    def __init__(self, input_channels, output_channels):
        super().__init__()
        #define the operations inside the block
        self.block = nn.Sequential(
          nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1,bias=False),
          nn.BatchNorm2d(output_channels),
          nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1,bias=False),
          nn.BatchNorm2d(output_channels),
          nn.ReLU(inplace=True)
        )

    def forward(self,x):
      x = self.block(x)
      return x
      
class downsampling_block(nn.Module):
  def __init__(self, input_channels, output_channels):
    """
        Initializes the downsampling block.

        Args:
            input_channels (int): The number of channels in the input feature map.
            output_channels (int): The number of channels after the convolution block.
    """
    super().__init__()
    self.conv_block = convolution_block(input_channels, output_channels)
    self.pool = nn.MaxPool2d((2,2))

  def forward(self,x):
    x = self.conv_block(x)
    p = self.pool(x)
    # Return both the pre-pooled feature map for the skip connection
    # and the pooled feature map for the next layer in the encoder.
    return x,p 

class upsampling_block(nn.Module):
  """
    An upsampling block for the U-Net expansive path.
    It combines upsampling with a skip connection from the encoder path.
  """
  def __init__(self, input_channels, output_channels):
    super().__init__()
    #define the up-convolution
    self.up_convolution = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2, padding=0)
    #define the convolution: current output + skip connection
    self.convolution_block = convolution_block(
                                               output_channels+output_channels, #current + skip connection
                                               output_channels
                                               )
  def forward(self,x,skip_connection):
    x = self.up_convolution(x)
    #concatenate the previous result with the skipp connection before applying the convolution block
    x = torch.cat([x,skip_connection],axis=1)
    #apply the convolution as always
    x = self.convolution_block(x)
    return x
    
class UNET_Architecture(nn.Module):
  def __init__(self):
    super().__init__()
    '''
      CONTRACTING PATH
    '''
    self.encoder_1 = downsampling_block(3,64)
    self.encoder_2 = downsampling_block(64,128)
    self.encoder_3 = downsampling_block(128,256)
    self.encoder_4 = downsampling_block(256,512)

    '''
      BOTTLENECK
    '''
    self.bottleneck = convolution_block(512,1024)

    '''
      EXPANSIVE PATH
    '''
    self.decoder_1 = upsampling_block(1024,512)
    self.decoder_2 = upsampling_block(512,256)
    self.decoder_3 = upsampling_block(256,128)
    self.decoder_4 = upsampling_block(128,64)

    '''
      OUTPUT LAYER
    '''
    self.output_layer = nn.Conv2d(64,1,kernel_size=1)
  
  def forward(self,x):
    #contracting path
    x1,p1 = self.encoder_1(x)
    x2,p2 = self.encoder_2(p1)
    x3,p3 = self.encoder_3(p2)
    x4,p4 = self.encoder_4(p3)

    #bottleneck
    b = self.bottleneck(p4)

    #expanding path
    d1 = self.decoder_1(b,x4) #use of skip connection
    d2 = self.decoder_2(d1,x3) #use of skip connection
    d3 = self.decoder_3(d2,x2) #use of skip connection
    d4 = self.decoder_4(d3,x1) #use of skip connection

    #output layer
    output = self.output_layer(d4)

    return output