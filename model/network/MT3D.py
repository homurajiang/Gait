import torch
import torch.nn as nn
import numpy as np

from .basic_blocks import SetBlock, BasicConv2d, M3DPooling, FramePooling, FramePooling1, LocalTransform, BasicConv3DB, GMAP, SeparateFC


class MTNet(nn.Module):
    def __init__(self, hidden_dim):
        super(MTNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        _set_in_channels = 1
        _set_channels = [32, 64, 128, 128]
        self.layer1 = nn.Conv3d(_set_in_channels, _set_channels[0], kernel_size=(3,3,3), stride=(2,1,1), padding=1)

        # Transform clip 每个clip分开卷
        self.local_transform1 = LocalTransform(_set_channels[0], _set_channels[0],s=3)
        self.B3D_layer2_S = BasicConv3DB(_set_channels[0], _set_channels[1], padding=1)
        self.M3D_layer2_S = M3DPooling()
        self.B3D_layer2_L = BasicConv3DB(_set_channels[0], _set_channels[1], padding=1)
        self.M3D_layer2_L = M3DPooling()

        self.local_transform2 = LocalTransform(_set_channels[1], _set_channels[1],s=3)
        self.B3D_layer3_S1 = BasicConv3DB(_set_channels[1], _set_channels[2], padding=1)
        self.B3D_layer3_S2 = BasicConv3DB(_set_channels[2], _set_channels[3], padding=1)
        self.B3D_layer3_L1 = BasicConv3DB(_set_channels[1], _set_channels[2], padding=1)
        self.B3D_layer3_L2 = BasicConv3DB(_set_channels[2], _set_channels[3], padding=1)

        self.local_transform3 = LocalTransform(_set_channels[3], _set_channels[3],s=3)
        self.framepooling_S = FramePooling1()
        self.framepooling_L = FramePooling1()
        self.gmap_S = GMAP(w=22)
        self.gmap_L = GMAP(w=22)

        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(64, _set_channels[3], self.hidden_dim)))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, silho, batch_frame=None):
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2) #[12, 30, 1, 64, 44]
        del silho

        import pdb
        # pdb.set_trace()
        
        #layer1
        x = self.layer1(x.permute(0,2,1,3,4).contiguous()) #output [12, 32, 15, 64, 44]
        x1 = self.local_transform1(x)           # [12, 32, 5, 64, 44]
        #layer2
        
        x = self.B3D_layer2_S(x)                # [12, 64, 15, 64, 44]
        x = self.M3D_layer2_S(x)                # [12, 64, 15, 32, 22]
        x1 = self.B3D_layer2_L(x1)              # [12, 64, 5, 64, 44]
        x1 = self.M3D_layer2_L(x1)              # [12, 64, 5, 32, 22]

        x1 = x1 + self.local_transform2(x)      # [12, 64, 5, 32, 22]
        #layer3
        x = self.B3D_layer3_S1(x)
        x = self.B3D_layer3_S2(x)               # [12, 128, 15, 32, 22]
        x1 = self.B3D_layer3_L1(x1)
        x1 = self.B3D_layer3_L2(x1)             # [12, 128, 5, 32, 22]
        x1 = x1 + self.local_transform3(x)      # [12, 128, 5, 32, 22]
        #Framepooling & GAP GMP
        x = self.framepooling_S(x)              # [12, 128, 1, 32, 22]
        x = self.gmap_S(x)                      # [12, 128, 1, 32, 1]
        x1 = self.framepooling_L(x1)            
        x1 = self.gmap_L(x1)


        #Separate FC
        feature = torch.cat((x,x1),dim=3)               # [12, 128, 1, 64, 1]
        del x1
        del x
        # x = self.fc(x)

        feature = feature.squeeze(-1)                    # [12, 128, 1, 64]
        feature = feature.permute(0, 3, 2, 1).contiguous() # [12, 64, 1, 128]
        feature = feature.matmul(self.fc_bin)

        return feature.squeeze(2), None # [12, 64, 128]