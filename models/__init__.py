from __future__ import absolute_import

from models.cbam_resnext import cbam_resnext29_8x64d, cbam_resnext29_16x64d
from models.genet import ge_resnext29_8x64d, ge_resnext29_16x64d
from models.shake_shake import shake_resnet26_2x32d, shake_resnet26_2x64d
from models.sknet import sk_resnext29_16x32d, sk_resnext29_16x64d
from models.squeezenet import squeezenet
from models.stochasticdepth import (stochastic_depth_resnet18,
                                    stochastic_depth_resnet34,
                                    stochastic_depth_resnet50,
                                    stochastic_depth_resnet101,
                                    stochastic_depth_resnet152)
from models.xception import xception

from .attention import *
from .densenet import *
from .dla import *
from .dpn import DPN26
from .dynamic_resnet20 import *
from .efficientnetb0 import *
from .googlenet import *
from .inceptionv3 import *
from .inceptionv4 import *
from .lenet import *
from .masked_resnet20 import *
from .mobilenet import *
from .mobilenetv2 import *
from .nasnet import *
from .pnasnet import *
from .preact_resnet import *
from .regnet import *
from .resnet import *
from .resnet20 import *
from .resnext import *
from .rir import *
from .sample_resnet20 import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .slimmable_resnet20 import *
from .stochasticdepth import *
from .supernet import *
from .vgg import *
from .wideresidual import *
from .xception import *

__model_factory = {
    'dynamic': dynamic_resnet20,# 参考once for all那版
    'masked': masked_resnet20, #基于mask进行实现
    'resnet20': resnet20,
    'sample': sample_resnet20,
    'slimmable': slimmable_resnet20, #最开始那版slimmable network
    'super': SuperNet,
    'densenet': densenet_cifar,
    'senet': senet18_cifar,
    'googlenet': GoogLeNet,
    'dla': DLA,
    'shufflenet': ShuffleNetG2,
    'shufflenetv2': ShuffleNetV2,
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'efficientnetb0': EfficientNetB0,
    'lenet': LeNet,
    'mobilenet': MobileNet,
    'mobilenetv2': MobileNetV2,
    'pnasnet': PNASNetB,
    'preact_resnet': PreActResNet18,
    'regnet': RegNetX_200MF,
    'resnext': ResNeXt29_2x64d,
    'vgg': vgg11,
    'attention56': attention56,
    'attention92': attention92,
    'inceptionv3': inceptionv3,
    'inceptionv4': inceptionv4,
    'inception_resnet_v2': inception_resnet_v2,
    'nasnet': nasnet,
    'rir': resnet_in_resnet,
    'squeezenet': squeezenet,
    'stochastic_depth_resnet18': stochastic_depth_resnet18,
    'stochastic_depth_resnet34': stochastic_depth_resnet34,
    'stochastic_depth_resnet50': stochastic_depth_resnet50,
    'stochastic_depth_resnet101': stochastic_depth_resnet101,
    'stochastic_depth_resnet152': stochastic_depth_resnet152,
    'wideresnet': wideresnet,
    'xception': xception,
    'dpn': DPN26,
    'shake_resnet26_2x32d': shake_resnet26_2x32d,
    'shake_resnet26_2x64d': shake_resnet26_2x64d,
    'ge_resnext29_8x64d': ge_resnext29_8x64d,
    'ge_resnext29_16x64d': ge_resnext29_16x64d,
    'sk_resnext29_16x32d': sk_resnext29_16x32d,
    'sk_resnext29_16x64d': sk_resnext29_16x64d,
    'cbam_resnext29_16x64d': cbam_resnext29_16x64d,
    'cbam_resnext29_8x64d': cbam_resnext29_8x64d

}


def show_available_models():
    """Displays available models

    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes=10):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](num_classes=num_classes)
