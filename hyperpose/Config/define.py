from enum import Enum

class BACKBONE(Enum):
    # TODO(JZ)BACKBONE
    Default=0
    Mobilenetv1=1
    Vgg19=2
    Resnet18=3
    Resnet50=4
    Vggtiny=5
    Mobilenetv2=6
    Vgg16=7

class MODEL(Enum):
    # Openpose = 1
    # LightweightOpenpose = 0
    Openpose=0
    LightweightOpenpose=1
    PoseProposal=2
    MobilenetThinOpenpose=3

class DATA(Enum):
    MSCOCO=0
    MPII=1
    USERDEF=2
    MULTIPLE=3

class TRAIN(Enum):
    Single_train=0
    Parallel_train=1

class KUNGFU(Enum):
    Sync_sgd=0
    Sync_avg=1
    Pair_avg=2

class OPTIM(Enum):
    Adam=0
    SGD=2
    RMSprop=1
