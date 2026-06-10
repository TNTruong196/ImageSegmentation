import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

def conv3x3 (in_channels, out_channels, stride=1, padding =1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride =stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode ='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride =2
        )
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor =2, align_corners=True),
            conv1x1(in_channels, out_channels)
        )
    
def conv1x1 (in_channels, out_channels, groups = 1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1
    )


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling = True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1=conv3x3(in_channels, out_channels)
        self.conv2=conv3x3(out_channels, out_channels)
        if self.pooling :
            self.pool = nn.MaxPool2d(kernel_size=2, stride =2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool =x
        if self.pooling:
            x=self.pool(x)
        return x, before_pool
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(in_channels, out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1=conv3x3(2*self.out_channels, self.out_channels)
        else:
            self.conv1=conv3x3(self.out_channels, self.out_channels)
        self.conv2=conv3x3(self.out_channels, self.out_channels)
    
    def forward (self, from_down, from_up):
        from_up=self.upconv(from_up)
        if self.merge_mode == 'concat':
            x=torch.cat((from_up,from_down),1)
        else:
            x= from_up+ from_down
        x= F.relu(self.conv1(x))
        x= F.relu(self.conv2(x))
        return x
    
class UNet(nn.Module):
    
    def __init__(self, num_classes, in_channels=3, depth=5, start_filts = 64,
                up_mode='transpose', merge_mode='concat'):
        super(UNet,self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError(f'"{up_mode}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.')
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError(f'"{merge_mode}" is not a valid mode for merging up and down paths. Only "concat" and "add" are allowed.')

        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError('up_mode "upsample" is incompatible with merge_mode "add" at the moment.')
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth=depth
        self.start_filts=start_filts

        down_convs = []
        up_convs = []
        outs = self.start_filts
        for i in range(depth):
            ins=self.in_channels if i==0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i< depth - 1 else False
            down_conv = DownConv(ins,outs, pooling = pooling)
            down_convs.append(down_conv)

        for i in range(depth - 1):
            ins=outs
            outs=ins//2
            up_conv=UpConv(ins,outs,merge_mode=merge_mode, up_mode=up_mode)
            up_convs.append(up_conv)
        self.conv_final = conv1x1(outs, self.num_classes)

        self.down_convs = nn.ModuleList(down_convs)
        self.up_convs = nn.ModuleList(up_convs)

        self.reset_params()
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def reset_params (self):
        for i in self.modules():
            self.weight_init(i)
    
    def forward(self, x):
        encoder_outs = []

        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
        
        for i, module in enumerate(self.up_convs):
            before_pool=encoder_outs[-(i+2)]
            x= module(before_pool,x)

        x = self.conv_final(x)
        return x
    


if __name__ == "__main__": # Đoạn mã này chỉ chạy khi bạn thực thi trực tiếp file này.
    # Khởi tạo một đối tượng mạng UNet thực tế với cấu hình đầu ra gồm 3 nhóm đối tượng (classes).
    model = UNet(num_classes=3, depth=5, merge_mode='concat')
    
    # Tạo một mảng NumPy ngẫu nhiên mô phỏng 1 bức ảnh đầu vào (Batch size=1, Kênh màu RGB=3, Cao=320px, Rộng=320px).
    # Chuyển mảng này thành một Float Tensor của PyTorch.
    x = torch.from_numpy(np.random.random((1, 3, 320, 320)).astype(np.float32))
    x.requires_grad_() # Bật tính năng theo dõi tính đạo hàm cho Tensor đầu vào x.
    
    out = model(x) # Truyền ảnh x qua mạng (Lan truyền xuôi - Forward pass) để nhận kết quả đầu ra `out`.
    loss = torch.sum(out) # Định nghĩa một hàm Loss giả định bằng cách tính tổng tất cả các pixel đầu ra.
    loss.backward() # Lan truyền ngược (Backward pass) để tính toán ma trận Gradient (đạo hàm) cho toàn bộ trọng số của mạng.
    
    print("Lan truyền xuôi và ngược hoàn thành mượt mà trên bản PyTorch mới!") # In thông báo thành công ra màn hình.
