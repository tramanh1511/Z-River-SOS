from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3, stride=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        filters = [in_channels] + filters
        print(filters)
        for i in range(1, len(filters)):
            self.layers.append(ConvBlock(filters[i-1], filters[i], kernel_size, stride, dropout=dropout))
    
    def forward(self, x):
        enc_outs = []
        for layer in self.layers:
            x = layer(x)
            enc_outs.append(x)
            x = self.pool(x)
        return enc_outs, x

class Decoder(nn.Module):
    def __init__(self, filters, kernel_size=3, stride=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        
        for i in range(len(filters)-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(filters[i], filters[i-1], kernel_size=2, stride=2))
            self.dec_layers.append(ConvBlock(filters[i], filters[i-1], kernel_size, stride, dropout=dropout))
    
    def forward(self, x, enc_outs):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat((enc_outs[i], x), dim=1)
            x = self.dec_layers[i](x)
        return x


class RiverUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[64, 128, 256, 512], kernel_size=3, stride=1, dropout=0.1, cls=False):
        super(RiverUNet, self).__init__()
        self.encoder = Encoder(in_channels, filters, kernel_size, stride, dropout=dropout)
        self.bottleneck = ConvBlock(filters[-1], 2 * filters[-1], kernel_size, stride, dropout=dropout)
        self.decoder = Decoder(filters + [2 * filters[-1]], kernel_size, stride, dropout=dropout)
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.classifier = nn.Linear(2 * filters[-1], out_channels) if cls else None

    def forward(self, x, threshold: Optional[float] = None):
        enc_outs, x = self.encoder(x)
        # print([p.shape for p in enc_outs])
        x = self.bottleneck(x)
        if self.classifier:
            cls_input = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
            cls_output = self.classifier(cls_input)
        else: 
            cls_output = None
        x = self.decoder(x, enc_outs[::-1])
        x = self.final_conv(x)
        if threshold: 
            if self.classifier:
                pred = torch.where(F.sigmoid(cls_output) > threshold, 0, 1e5)
                return x - pred[:, None, None]
            else:
                return x
        return {'logit': x, 'cls_logit': cls_output} if self.classifier else {'logit': x}


############### Attention Block #######################################

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionDecoder(nn.Module):
    def __init__(self, filters, kernel_size=3, stride=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in range(len(filters)-1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(filters[i], filters[i-1], kernel_size=2, stride=2, dropout=dropout))
            self.attention_gates.append(AttentionGate(F_g=filters[i-1], F_l=filters[i-1], F_int=filters[i-1]//2))
            self.dec_layers.append(ConvBlock(filters[i], filters[i-1], kernel_size, stride, dropout=dropout))
    
    def forward(self, x, enc_outs):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            att = self.attention_gates[i](x, enc_outs[i])
            x = torch.cat((att, x), dim=1)
            x = self.dec_layers[i](x)
        return x

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[64, 128, 256, 512], kernel_size=3, stride=1):
        super(AttentionUNet, self).__init__()
        self.encoder = Encoder(in_channels, filters, kernel_size, stride)
        self.bottleneck = ConvBlock(filters[-1], filters[-1]*2, kernel_size, stride)
        self.decoder = AttentionDecoder(filters[::-1], kernel_size, stride)
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        enc_outs, x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, enc_outs[::-1])
        x = self.final_conv(x)
        return x

if __name__=="__main__":
    b, c, h, w = 4, 3, 256, 256
    x = torch.randn([b, c, h, w])
    model = RiverUNet(3, 1, filters=[16, 32, 64, 128, 256], kernel_size=3, cls = True)
    x = x.to('cuda:0')
    model = model.to('cuda:0')
    outputs = model(x)
    print(outputs.keys(), outputs['logit'].shape, outputs['cls_logit']
    )