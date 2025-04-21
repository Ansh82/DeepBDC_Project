import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.distributions import Bernoulli

##############################################
#          Efficient ResNet Model            #
##############################################

def init_layer(L):
    """Initialize layer weights using fan-in."""
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    """Flatten layer to convert 2D feature maps to 1D features."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class EfficientBlock(nn.Module):
    """Efficient ResNet block with depthwise separable convolutions."""
    
    def __init__(self, indim, outdim, half_res=False):
        super(EfficientBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        
        # Depthwise separable convolution instead of standard 3x3 conv
        self.C1_depthwise = nn.Conv2d(indim, indim, kernel_size=3, stride=2 if half_res else 1, 
                                     padding=1, groups=indim, bias=False)
        self.C1_pointwise = nn.Conv2d(indim, outdim, kernel_size=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        
        # Second depthwise separable convolution
        self.C2_depthwise = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, 
                                     groups=outdim, bias=False)
        self.C2_pointwise = nn.Conv2d(outdim, outdim, kernel_size=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)
        
        # Activation with ReLU6 for better quantization
        self.relu = nn.ReLU6(inplace=True)
        
        self.parametrized_layers = [self.C1_depthwise, self.C1_pointwise, self.BN1, 
                                   self.C2_depthwise, self.C2_pointwise, self.BN2]
        
        self.half_res = half_res
        
        # Shortcut connection
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
            
        for layer in self.parametrized_layers:
            init_layer(layer)
    
    def forward(self, x):
        # First depthwise separable convolution
        out = self.C1_depthwise(x)
        out = self.C1_pointwise(out)
        out = self.BN1(out)
        out = self.relu(out)
        
        # Second depthwise separable convolution
        out = self.C2_depthwise(out)
        out = self.C2_pointwise(out)
        out = self.BN2(out)
        
        # Shortcut connection
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu(out)
        
        return out

class EfficientBottleneckBlock(nn.Module):
    """Efficient bottleneck block with expansion and depthwise separable convolutions."""
    
    def __init__(self, indim, outdim, half_res=False, expansion=4):
        super(EfficientBottleneckBlock, self).__init__()
        
        self.indim = indim
        self.outdim = outdim
        bottleneckdim = int(outdim / expansion)
        
        # 1x1 conv for reducing dimensions
        self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
        self.BN1 = nn.BatchNorm2d(bottleneckdim)
        
        # Depthwise separable conv instead of 3x3 conv
        self.C2_depthwise = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, 
                                     stride=2 if half_res else 1, padding=1,
                                     groups=bottleneckdim, bias=False)
        self.C2_pointwise = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=1, bias=False)
        self.BN2 = nn.BatchNorm2d(bottleneckdim)
        
        # 1x1 conv for expanding dimensions
        self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
        self.BN3 = nn.BatchNorm2d(outdim)
        
        # Activation with ReLU6 for better quantization
        self.relu = nn.ReLU6(inplace=True)
        
        self.parametrized_layers = [self.C1, self.BN1, self.C2_depthwise, 
                                   self.C2_pointwise, self.BN2, self.C3, self.BN3]
        
        self.half_res = half_res
        
        # Shortcut connection
        if indim != outdim:
            self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
            
        for layer in self.parametrized_layers:
            init_layer(layer)
    
    def forward(self, x):
        # Shortcut connection
        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        
        # Bottleneck path
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        
        out = self.C2_depthwise(out)
        out = self.C2_pointwise(out)
        out = self.BN2(out)
        out = self.relu(out)
        
        out = self.C3(out)
        out = self.BN3(out)
        
        # Residual connection
        out = out + short_out
        out = self.relu(out)
        
        return out

class EfficientResNet(nn.Module):
    """Efficient ResNet implementation with depthwise separable convolutions."""
    
    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=False):
        super(EfficientResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        
        # Use more efficient conv for the first layer (smaller kernel)
        conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU6(inplace=True)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        init_layer(conv1)
        init_layer(bn1)
        
        trunk = [conv1, bn1, relu, pool1]
        
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0) and i != 3
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
        
        if flatten:
            avgpool = nn.AdaptiveAvgPool2d(1)  # More efficient than fixed size pooling
            trunk.append(avgpool)
            trunk.append(Flatten())
        
        self.feat_dim = [list_of_out_dims[3], 14, 14]  # Keeping same feature dimensions
        self.trunk = nn.Sequential(*trunk)
    
    def forward(self, x):
        return self.trunk(x)

def EfficientResNet10(flatten=True):
    return EfficientResNet(EfficientBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)

def EfficientResNet18(flatten=False):
    return EfficientResNet(EfficientBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)

def EfficientResNet34(flatten=True):
    return EfficientResNet(EfficientBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)

def EfficientResNet50(flatten=True):
    return EfficientResNet(EfficientBottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)

def EfficientResNet101(flatten=True):
    return EfficientResNet(EfficientBottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)

##############################################
#     Efficient ResNet Variant Model         #
##############################################

class EfficientSELayer(nn.Module):
    """Efficient Squeeze-and-Excitation layer with reduced parameters."""
    
    def __init__(self, channel, reduction=16):
        super(EfficientSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Use smaller SE layer with fewer parameters
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class EfficientDropBlock(nn.Module):
    """Efficient DropBlock implementation."""
    
    def __init__(self, block_size):
        super(EfficientDropBlock, self).__init__()
        self.block_size = block_size
    
    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            
            # Create mask
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).to(x.device)
            
            # Compute block mask
            block_mask = self._compute_block_mask(mask)
            
            # Normalization
            countM = block_mask.numel()
            count_ones = block_mask.sum()
            
            # Avoid division by zero
            if count_ones > 0:
                return block_mask * x * (countM / count_ones)
            return x
        else:
            return x
    
    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]
        
        # Create offsets
        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).repeat(self.block_size),
            ]
        ).t().to(mask.device)
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).to(mask.device).long(), offsets.long()), 1)
        
        # Apply offsets to indices
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size**2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            block_idxs = non_zero_idxs + offsets
            
            # Pad mask and set values
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        
        # Return block mask
        return 1 - padded_mask

class EfficientBasicBlock(nn.Module):
    """Efficient basic block with depthwise separable convolutions."""
    
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, 
                 drop_block=False, block_size=1, use_se=False):
        super(EfficientBasicBlock, self).__init__()
        
        # Depthwise separable convolutions
        self.conv1_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, groups=inplanes, bias=False)
        self.conv1_pw = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2_dw = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes, bias=False)
        self.conv2_pw = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3_dw = nn.Conv2d(planes, planes, kernel_size=3, padding=1, groups=planes, bias=False)
        self.conv3_pw = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # Use ReLU6 for better quantization support
        self.relu = nn.ReLU6(inplace=True)
        
        # Efficient MaxPool with stride
        self.maxpool = nn.MaxPool2d(stride) if stride > 1 else nn.Identity()
        
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = EfficientDropBlock(block_size=self.block_size)
        
        # Squeeze-and-Excitation
        self.use_se = use_se
        if self.use_se:
            self.se = EfficientSELayer(planes, 4)
    
    def forward(self, x):
        self.num_batches_tracked += 1
        
        residual = x
        
        # First convolution block
        out = self.conv1_dw(x)
        out = self.conv1_pw(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2_dw(out)
        out = self.conv2_pw(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Third convolution block
        out = self.conv3_dw(out)
        out = self.conv3_pw(out)
        out = self.bn3(out)
        
        # Squeeze-and-Excitation
        if self.use_se:
            out = self.se(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        # Dropout and DropBlock
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        
        return out

class EfficientResNetVariant(nn.Module):
    """Efficient ResNet variant with depthwise separable convolutions."""
    
    def __init__(self, block, n_blocks, keep_prob=1.0, avg_pool=False, drop_rate=0.0,
                 dropblock_size=5, num_classes=-1, use_se=False):
        super(EfficientResNetVariant, self).__init__()
        
        self.inplanes = 3
        self.use_se = use_se
        
        # Layers with reduced parameters
        self.layer1 = self._make_layer(block, n_blocks[0], 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, n_blocks[1], 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, n_blocks[2], 320, stride=2, drop_rate=drop_rate,
                                     drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, n_blocks[3], 640, stride=1, drop_rate=drop_rate,
                                     drop_block=True, block_size=dropblock_size)
        
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.feat_dim = [640, 10, 10]
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Optional classifier
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(640, self.num_classes)
    
    def _make_layer(self, block, n_block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        if n_block == 1:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, self.use_se)
        else:
            layer = block(self.inplanes, planes, stride, downsample, drop_rate, self.use_se)
        layers.append(layer)
        self.inplanes = planes * block.expansion
        
        for i in range(1, n_block):
            if i == n_block - 1:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, drop_block=drop_block,
                              block_size=block_size, use_se=self.use_se)
            else:
                layer = block(self.inplanes, planes, drop_rate=drop_rate, use_se=self.use_se)
            layers.append(layer)
        
        return nn.Sequential(*layers)
    
    def forward(self, x, is_feat=False):
        # Forward pass through layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def EfficientResNet12(keep_prob=1.0, avg_pool=True, **kwargs):
    """Constructs an Efficient ResNet-12 model."""
    model = EfficientResNetVariant(EfficientBasicBlock, [1, 1, 1, 1], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model

def EfficientResNet34s(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs an Efficient ResNet-34s model."""
    model = EfficientResNetVariant(EfficientBasicBlock, [2, 3, 4, 2], keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model