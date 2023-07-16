import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.route import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, p=None, p_w=None, p_a=None, info=None, clip_threshold=1e10, LU = False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.dropout = torch.nn.Dropout(0.3)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgp2d = nn.AvgPool2d(8)
        self.ID_mat = torch.eye(num_classes).cuda()

        # self.fc = nn.Linear(nChannels[3], num_classes, bias=False)
        if p is None or info is None:
            self.fc = nn.Linear(nChannels[3], num_classes)
        else:
            if LU:
                print('use LINE')
                self.fc = RouteLUNCH(nChannels[3], num_classes, p_w=p_w, p_a=p_a, info=info, clip_threshold = clip_threshold)
            else:
                print('use dice')
                self.fc = RouteDICE(nChannels[3], num_classes, p=p, info=info)
     
        # self.fc = nn.Linear(nChannels[3], num_classes)
        # self.fc.weight.requires_grad = False        # Freezing the weights during training
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.orthogonal(m.weight.data)   # Initializing with orthogonal rows
                m.bias.data.zero_()

    def features(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.relu(self.bn1(out))
        return out
    
    def forward_pool_feat(self, feat):
        feat = self.relu(self.bn1(feat))
        out = F.avg_pool2d(feat, 8)
        out = out.view(-1, self.nChannels)

        # out = F.normalize(out, dim=1, p=2)
        # out = torch.abs(self.fc(out))
        out = self.fc(out)
        return out

    def forward_threshold_features(self, x, threshold=1e10):
        feat = self.features(x)
        feat = self.relu(self.bn1(feat))
        out = F.avg_pool2d(feat, 8)
        feat = feat.clip(max=threshold)
        out = out.view(-1, self.nChannels)  
        # out = F.normalize(out, dim=1, p=2)
        return out

    def forward_features(self, x):
        feat = self.features(x)
        feat = self.relu(self.bn1(feat))
        out = F.avg_pool2d(feat, 8)
        out = out.view(-1, self.nChannels)  
        # out = F.normalize(out, dim=1, p=2)
        return out
    
    
    def forward_head(self, feat):
        
        # out = torch.abs(self.fc(out))
        out = self.fc(feat)
        return out
    
    def forward(self, x, fc_params=None):
        feat = self.features(x)
        feat = self.relu(self.bn1(feat))
        # out = self.relu(self.bn1(out))
        # out = self.relu(out)

        out = F.avg_pool2d(feat, 8)
        out = out.view(-1, self.nChannels)
        # print(out.size(), out_feat.size())

        # out = F.normalize(out, dim=1, p=2)     
        # print(out.size())
        # out = torch.abs(self.fc(out))
        out = self.fc(out)
        return out

    def forward_threshold(self, x, threshold=1e10):
        feat = self.features(x)
        feat = self.relu(self.bn1(feat))

        out = F.avg_pool2d(feat, 8)
        out = out.clip(max=threshold)
        out = out.view(-1, self.nChannels)
        
        out = self.fc(out)
        return out

    def forward_LINE(self, x, threshold=1e10):
        out = self.features(x)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.clip(max=threshold)
        out = out.view(-1, self.nChannels)
        out, feat = self.fc(out)
        return out, feat

    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        # out_list.append(out)
        out = self.block1(out)
        # out_list.append(out)
        out = self.block2(out)
        # out_list.append(out)
        out = self.block3(out)
        # out_list.append(out)
        
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        # out = F.normalize(out, dim=1, p=2)
        y = self.fc(out)
        return y, out_list
    
    def intermediate_forward(self, x, layer_index=1):
        out = self.conv1(x)
        # if layer_index == 1:
        #     out = self.layer1(out)
        # elif layer_index == 2:
        #     out = self.block1(out)
        #     out = self.block2(out)
        # elif layer_index == 3:
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out
    
    def _forward(self, x):
        self.activations = []
        self.gradients = []
        self.zero_grad()
        
        out = self.features(x)
        out = self.relu(self.bn1(out))
        out = self.avgp2d(out)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out

    # LINE 
    def remove_handles(self):
        for handle in self.handles_list:
            handle.remove()
        self.handles_list.clear()
        self.activations = []
        self.gradients = []
    # LINE 
    def _compute_taylor_scores(self, inputs, labels):
        self._hook_layers()
        outputs = self._forward(inputs)
        outputs[0, labels.item()].backward(retain_graph=True)

        first_order_taylor_scores = []
        self.gradients.reverse()

        for i, layer in enumerate(self.activations):
            first_order_taylor_scores.append(torch.mul(layer, self.gradients[i]))
                
        self.remove_handles()
        return first_order_taylor_scores, outputs
    # LINE 
    def _hook_layers(self):
        def backward_hook_relu(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].to(self.device))

        def forward_hook_relu(module, input, output):
            # mask output by pruned_activations_mask
            # In the first model(input) call, the pruned_activations_mask
            # is not yet defined, thus we check for emptiness
            if self.pruned_activations_mask:
              output = torch.mul(output, self.pruned_activations_mask[len(self.activations)].to(self.device)) #+ self.pruning_biases[len(self.activations)].to(self.device)
            self.activations.append(output.to(self.device))
            return output

        for module in self.modules():
            if isinstance(module, nn.AvgPool2d):
                self.handles_list.append(module.register_forward_hook(forward_hook_relu))
                self.handles_list.append(module.register_backward_hook(backward_hook_relu))

def WideResNet28(**kwargs):
    return WideResNet(depth=28, widen_factor=10, **kwargs)