import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)
class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()
class YOLOLayer(nn.Module):
    def __init__(self, channels, anchors, num_classes=20, img_dim=416):
        super().__init__()
        self.anchors = anchors # three anchors per YOLO Layer
        self.num_anchors = len(anchors) # 3
        self.num_classes = num_classes # VOC classes 20
        self.img_dim = img_dim # 입력 이미지 크기 416
        self.grid_size = 0

        # 예측을 수행하기 전, smooth conv layer 입니다.
        self.conv = nn.Sequential(
            BasicConv(channels, channels*2, 3, stride=1, padding=1),
            nn.Conv2d(channels*2, 75, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)

        # prediction
        # x: batch, channels, W, H
        batch_size = x.size(0)
        grid_size = x.size(2) # S = 13 or 26 or 52
        device = x.device

        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5,
                            grid_size, grid_size) # shape = (batch, 3, 25, S, S)
        
        # shape change (batch, 3, 25, S, S) -> (batch, 3, S, S, 25)
        prediction = prediction.permute(0, 1, 3, 4, 2)
        prediction = prediction.contiguous()

        obj_score = torch.sigmoid(prediction[..., 4]) # Confidence: 1 if object, else 0
        pred_cls = torch.sigmoid(prediction[..., 5:]) # 바운딩 박스 좌표

        # grid_size 갱신
        if grid_size != self.grid_size:
            # grid_size를 갱신하고, transform_outputs 함수를 위해 anchor 박스를 전처리 합니다.
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # calculate bounding box coordinates
        pred_boxes = self.transform_outputs(prediction)

        # output shape(batch, num_anchors x S x S, 25)
        # ex) at 13x13 -> [batch, 507, 25], at 26x26 -> [batch, 2028, 25], at 52x52 -> [batch, 8112, 25]
        # 최종적으로 YOLO는 10647개의 바운딩박스를 예측합니다.
        output = torch.cat((pred_boxes.view(batch_size, -1, 4),
                    obj_score.view(batch_size, -1, 1),
                    pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output


    # grid_size를 갱신하고, transform_outputs 함수를 위해 anchor 박스를 전처리 합니다.
    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size # ex) 13, 26, 52
        self.stride = self.img_dim / self.grid_size

        # cell index 생성
        # transform_outputs 함수에서 바운딩 박스의 x, y좌표를 예측할 때 사용합니다.
        # 1, 1, S, S
        self.grid_x = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).type(torch.float32)
        # 1, 1, S, S
        self.grid_y = torch.arange(grid_size, device=device).repeat(1, 1, grid_size, 1).transpose(3,2).type(torch.float32)

        # anchors를 feature map 크기로 정규화, [0~1] 범위
        scaled_anchors = [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        # tensor로 변환
        self.scaled_anchors = torch.tensor(scaled_anchors, device=device)

        # transform_outputs 함수에서 바운딩 박스의 w, h를 예측할 때 사용합니다.
        # shape=(3,2) -> (1, 3, 1, 1)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))


    # 예측한 바운딩 박스 좌표를 계산하는 함수입니다.
    def transform_outputs(self, prediction):
        # prediction = (batch, num_anchors, S, S, coordinates + classes)
        device = prediction.device
        x = torch.sigmoid(prediction[..., 0]) # sigmoid(box x), 예측값을 sigmoid로 감싸서 [0~1] 범위
        y = torch.sigmoid(prediction[..., 1]) # sigmoid(box y), 예측값을 sigmoid로 감싸서 [0~1] 범위
        w = prediction[..., 2] # 예측한 바운딩 박스 너비
        h = prediction[..., 3] # 예측한 바운딩 박스 높이

        pred_boxes = torch.zeros_like(prediction[..., :4]).to(device)
        pred_boxes[..., 0] = x.data + self.grid_x # sigmoid(box x) + cell x 좌표
        pred_boxes[..., 1] = y.data + self.grid_y # sigmoid(box y) + cell y 좌표
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        return pred_boxes * self.stride

def parse_cfg(cfgfile):
    '''
    configuration 파일을 입력으로 받습니다.
    
    blocks의 list를 반환합니다. 각 blocks는 신경망에서 구축되어지는 block을 의미합니다.
    block는 list안에 dictionary로 나타냅니다.
    '''
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')               # lines를 list로 저장합니다.
    lines = [x for x in lines if len(x) > 0]      # 빈 lines를 삭제합니다.
    lines = [x for x in lines if x[0] != '#']     # 주석을 삭제합니다.
    lines = [x.rstrip().lstrip() for x in lines]  # 공백을 제거합니다.
    
    # blocks를 얻기 위해 결과 list를 반복합니다.
    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':              # 새로운 block의 시작을 표시합니다.
            if len(block) != 0:         # block이 비어있지 않으면, 이전 block의 값을 저장합니다.
                blocks.append(block)    # 이것을 blocks list에 추가합니다.
                block = {}              # block을 초기화 합니다.
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

# parse_cfg 함수 잘 동작하는지 확인
# blocks = parse_cfg('D:\Object_Detection_Exe\cfg\yolo_v3.cfg')
# print(blocks)

def create_layers(blocks_list):
    hyperparams = blocks_list[0]
    channels_list = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()

    for layer_ind, layer_dict in enumerate(blocks_list[1:]):
        modules = nn.Sequential()

        if layer_dict['type'] == 'convolutional':
            filters = int(layer_dict['filters'])
            kernel_size = int(layer_dict['size'])
            pad = (kernel_size - 1) // 2
            bn = layer_dict.get('batch_normalize', 0)

            conv2d = nn.Conv2d(in_channels=channels_list[-1], out_channels=filters, kernel_size=kernel_size,
                               stride=int(layer_dict['stride']), padding=pad, bias=not bn)
            modules.add_module('conv_{0}'.format(layer_ind), conv2d)

            if bn:
                bn_layer = nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                modules.add_module('batch_norm_{0}'.format(layer_ind), bn_layer)
            
            if layer_dict['activation'] == 'leaky':
                activn = nn.LeakyReLU(0.1)
                modules.add_module('leky_{0}'.format(layer_ind), activn)

        elif layer_dict["type"] == "upsample":
            stride = int(layer_dict["stride"])
            upsample = nn.Upsample(scale_factor = stride)
            modules.add_module("upsample_{}".format(layer_ind), upsample) 

        elif layer_dict["type"] == "shortcut":
            backwards=int(layer_dict["from"])
            filters = channels_list[1:][backwards]
            modules.add_module("shortcut_{}".format(layer_ind), EmptyLayer())
            
        elif layer_dict["type"] == "route":
            layers = [int(x) for x in layer_dict["layers"].split(",")]
            filters = sum([channels_list[1:][l] for l in layers])
            modules.add_module("route_{}".format(layer_ind), EmptyLayer())

        elif layer_dict["type"] == "yolo":
            anchors = [int(a) for a in layer_dict["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            # ex) at 13x13, 'mask': '6,7,8'
            # mask는 anchors index를 의미합니다.
            # yolo layer당 3개의 anchors를 할당 합니다.
            # mask는 yolo layer feature map size에 알맞는 anchors를 설정합니다.
            mask = [int(m) for m in layer_dict["mask"].split(",")]
            
            anchors = [anchors[i] for i in mask] # 3 anchors
            
            num_classes = int(layer_dict["classes"]) # 20
            img_size = int(hyperparams["height"]) # 416 
            
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module("yolo_{}".format(layer_ind), yolo_layer)
            
        module_list.append(modules)       
        channels_list.append(filters)

    return hyperparams, module_list

