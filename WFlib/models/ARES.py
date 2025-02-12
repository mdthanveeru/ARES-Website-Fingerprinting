import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.layers import DropPath, Mlp

class TopM_MHSA(nn.Module):
    # Multipe layers of MHSA block
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()
        # initializes multiple layers of MHSA_Block in a sequential manner.
        self.nets = nn.ModuleList([MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x, pos_embed):
        output = x + pos_embed # Add positional embeddings to input features(provide information about the position of each token in the sequence)
        for layer in self.nets: # Pass through each MHSA_Block
            output = layer(output)
        return output  #output of one block becomes the input to the next block.
        
class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()
        
        self.num_heads = num_heads #8 heads
        head_dim = dim // num_heads #256/8=32 dim per head
        self.scale = head_dim ** -0.5 #1/sqrt(32) A scaling factor used to normalize the attention scores.
        self.top_m = top_m #Only the top 20 attention scores are used, and the rest are masked.

        self.qkv = nn.Linear(dim , dim*3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),#apply softmax function along the last dimension(across rows)
            nn.Dropout(dropout),#randomly sets elements to zero during training to prevent overfitting.
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim), #applies a learnable weight matrix and bias to transform input features y= x.Weight+bias
            nn.Dropout(dropout), #randomly sets elements to zero during training to prevent overfitting.
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):#initialize the weights and biases of different layers
        if isinstance(m, nn.Linear): #checks whether m is a Linear layer
            trunc_normal_(m.weight, std=.02) #set the weights to small random values
            if isinstance(m, nn.Linear) and m.bias is not None:# if the layer has bias
                nn.init.constant_(m.bias, 0) #set ias to 0
        elif isinstance(m, nn.LayerNorm): #if the module is nn.LayerNorm
            nn.init.constant_(m.bias, 0) #initializes bias=0
            nn.init.constant_(m.weight, 1.0) #wt=1
            #LayerNorm applies normalization and scaling. If the weight was randomly initialized, it could introduce unnecessary distortions in the input. Setting it to 1.0 ensures that the model starts without any artificial scaling.

    def forward(self, x):
        B, N, C = x.shape #B:Batch size(512) N:Sequence length(31) C:Number of channels(256).
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Linear projection to get Query (Q), Key (K), and Value (V) and Rearrange for multi-head attention
        q, k, v = qkv[0], qkv[1], qkv[2] ## Split Q, K, V

        attn = (q @ k.transpose(-2, -1)) * self.scale # Compute scaled attention scores
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)#This creates a tensor filled with zeros.
        #device=q.device=>ensures that the mask is created on the same device (CPU or GPU) as the q tensor
        #requires_grad=False=> don't need to calculate gradients
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        #finds the indices of the top top_m(20) largest values along the last dimension (dim=-1). The largest=True =>we want the largest values
        #[1]=> selects the output of torch.topk that contains the indices of the top k values
        mask.scatter_(-1, index, 1.)
        # places the value 1 into the mask tensor at the positions specified by index
        attn = torch.where(mask>0, attn, torch.full_like(attn, float('-inf')))
        #if the mask value is 0, it replaces the attn value with -inf.

        attn = self.attn_drop(attn)#applies the softmax and dropout operations defined above
        x = (attn @ v).transpose(1,2).reshape(B, N, C) #computes the dot product of the attention weights and the value tensor
        #then transpose and reshape with (B,N,C)=> shape=(1,4,6)
        x = self.proj_drop(x) #applies the self.proj_drop module to the output x
        return x

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m) #create an instance of TopMAttention
        self.drop_path = DropPath(drop_path_rate) #create an instance of DropPath(#randomly drops values)
        self.norm1 = nn.LayerNorm(embed_dim) #create an instance of LayerNorm(#Layer Normalization)
        self.norm2 = nn.LayerNorm(embed_dim) #(x - mean(x)) / sqrt(variance(x) + epsilon(small value to prevent div by 0)) * gamma + beta
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1) #create an instance of Mlp(#Multi-Layer Perceptron)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) #add the output of the attention module to the input x
        #Residual connections
        x = x + self.drop_path(self.mlp(self.norm2(x))) #add the output of the mlp module to the input x
        return x


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1): 
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"), 
            #padding="same" ensures that the output sequence length is the same as the input sequence length.
            nn.BatchNorm1d(out_channels), #Batch Normalization
            nn.ReLU(), # -ve values=> 0, +ve values=> no change
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),  
            nn.BatchNorm1d(out_channels), 
            nn.ReLU()
        )
        #Downsample=> to make both i/p and o/p dimensions same, so it can be added in residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            #initializes the weights of self.downsample with small random values from a normal distribution (mean = 0, std = 0.01).
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU() #apply ReLU activation one more time

    def forward(self, x): #x is a 3D tensor
        out = self.net(x) #forward pass
        res = x if self.downsample is None else self.downsample(x) #check if downsampling is required, then do it or not
        return self.last_relu(out + res) #residual connection


class LocalProfiling(nn.Module): 
    """ Local Profiling module in ARES """
    def __init__(self, in_channels=8): #8 is the number of features in the input sequence
        super(LocalProfiling, self).__init__() 
        
        self.net = nn.Sequential(
            #in (512,8,8000)
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7), #no of channels 8=>32
            # (512,32,8000)
            nn.MaxPool1d(kernel_size=8, stride=4), #downsample seq len by a factor of 4 (8000=>2000)
            # (512,32,2000)
            nn.Dropout(p=0.1), #Randomly sets 10% of values to 0. Scales up the remaining values by ≈1.11.
            # (512,32,2000)
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7), #no of channels 32=>64
            # (512,64,2000)
            nn.MaxPool1d(kernel_size=8, stride=4), #downsample seq len by a factor of 4 (2000=>500)
            # (512,64,500)
            nn.Dropout(p=0.1), 
            # (512,64,500)
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7), #no of channels 64=>128
            # (512,128,500)
            nn.MaxPool1d(kernel_size=8, stride=4), #downsample seq len by a factor of 4 (500=>125)
            # (512,128,125)
            nn.Dropout(p=0.1),
            # (512,128,125)
            ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7), #no of channels 128=>256
            # (512,256,125)
            nn.MaxPool1d(kernel_size=8, stride=4), #downsample seq len by a factor of 4 (125=>31)
            # (512,256,31)
            nn.Dropout(p=0.1),
            # (512,256,31)
        )
    def forward(self, x): #forward pass
        x = self.net(x) #input is passed through the above layers
        return x

class ARES(nn.Module):
    def __init__(self, num_classes,):
        super(ARES, self).__init__()
    #local variables
        embed_dim = 256 # Embedding dimension: The size of feature vectors for each token.
        num_heads = 8 # Number of attention heads in multi-head self-attention (MHSA).
        dim_feedforward = 256 * 4 # Hidden layer size in the feedforward network (typically 4x embed_dim).
        num_mhsa_layers = 4 # Number of MHSA (Multi-Head Self-Attention) layers in the model.
        dropout = 0.1 # Dropout rate to prevent overfitting-10%
        max_len = 29 # Maximum sequence length (number of time steps/tokens the model can process)
        top_m=20  # Number of top attention scores to keep in Top-M Attention
        in_channels = 8 # Number of input channels (features per time step).
    
    #attributes of this class
        self.profiling = LocalProfiling(in_channels) # initializes a LocalProfiling layer
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight #creates a positional embedding layer
        # Positional embeddings are used to provide the model with information about the position of elements in the sequence.
        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m) #initializes a custom Multi-Head Self-Attention (MHSA) layer
        self.mlp = nn.Linear(embed_dim, num_classes) #initializes a fully connected (linear) layer for the final classification.
        # that takes an input of dimension embed_dim(256) & projects it to dimension num_classes(100)
    
    #function of this class
    def forward(self, x): #defines how the input x is processed to produce the output.
        x = self.profiling(x) #input x is passed through the LocalProfiling layer 
        # (512,256,31)
        x = x.permute(0, 2, 1) #rearranges the dimensions of the tensor x. (batch_size, in_channels, sequence_length) it is permuted to (batch_size, sequence_length, in_channels) 
        # (512,31,256)
        x = self.topm_mhsa(x, self.pos_embed.unsqueeze(0)) #input x is passed through the TopM_MHSA layer
        #unsqueeze(0) adds a batch dimension to the positional embeddings, making their shape compatible with the input x for proper addition
        #Shape of self.pos_embed before unsqueeze: (29, 256) - (max_len, embed_dim)
        #Shape of self.pos_embed after unsqueeze: (1, 29, 256) - (1, max_len, embed_dim)
        x = x.mean(dim=1) #calculates the mean along dimension 1 (N), which represents the sequence length
        #Before: (512, 31, 256) - (B, N, C)
        #After: (512, 256) - (B, C)
        x = self.mlp(x) #Creates a linear layer that maps the final embedding to the output probabilities for each website class.
        #(512,100)  100 is our num_classes
        return x

if __name__ == '__main__':
    feat_len = 8000
    in_channels = 8
    net = ARES(num_classes=100)
    # print(net)
    x = np.random.rand(32, in_channels, feat_len)
    x = torch.tensor(x, dtype=torch.float32)
    out = net(x)
    print(f"in:{x.shape} --> out:{out.shape}")
    #prints just to ensure our model code is correct
