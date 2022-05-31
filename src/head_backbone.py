import torch as t
from torch import nn
from torchvision.models import vgg16,mobilenet_v2,resnet18

from src.region_proposal_network import RegionProposalNetwork
from src.head_detector import Head_Detector
from src.config import opt

def left_mob():
    #model = mobilenet_v2(pretrained=False,norm_layer=nn.InstanceNorm2d)
    model = mobilenet_v2(pretrained=True)
    features = list(model.features)[0:14]
    return nn.Sequential(*features)


def right_mob():
    #model = mobilenet_v2(pretrained=False,norm_layer=nn.InstanceNorm2d)
    model = mobilenet_v2(pretrained=True)
    features = list(model.features)[0:14]
    return nn.Sequential(*features)

def left_vgg16():
    model=vgg16(pretrained=False)
    #model.load_state_dict(t.load(opt.caffe_pretrain_path))
    features = list(model.features)[0:30]
    return nn.Sequential(*features)

def right_vgg16():
    model = vgg16(pretrained=False)
    #model.load_state_dict(t.load(opt.caffe_pretrain_path))
    features = list(model.features)[0:30]
    return nn.Sequential(*features)

def left_res():
    '''
    brainwash: resnet(pretrained=False,norm_layer=nn.InstanceNorm2d)
    restaurant: renet18(pretrained=True)
    '''
    model = resnet18(pretrained=True)
    features = [model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3]
    return nn.Sequential(*features)

def right_res():
    model = resnet18(pretrained=True)
    features = [model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3]
    return nn.Sequential(*features)

class mpsn(Head_Detector):
    """ Head detector based on VGG16 model.
    Have two components:
        1) Fixed feature extractor from the conv_5 layer of the VGG16
        2) A region proposal network on the top of the extractor.
    """
    feat_stride = 16

    def __init__(self, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):

        addnet = right_res()
        left_vgg = left_res()
        in_channel=256    #mob : 96    res : 256
        rpn = RegionProposalNetwork(
            in_channel, 256,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )
        super(mpsn, self).__init__(
            addnet,
            left_vgg,
            rpn
        )
    pass
