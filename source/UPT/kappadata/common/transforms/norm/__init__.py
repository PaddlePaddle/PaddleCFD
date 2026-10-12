from .kd_audioset_norm import KDAudioSetNorm
from .kd_cifar100_norm import KDCifar100Norm
from .kd_cifar10_norm import KDCifar10Norm
from .kd_esc50_norm import KDEsc50Norm
from .kd_image_net_norm import KDImageNetNorm
from .kd_sid_norm import KDSidNorm
from .kd_speech_ccommands_v1_norm import KDSpeechCommandsV1Norm
from .kd_speech_ccommands_v2_norm import KDSpeechCommandsV2Norm


def string_to_norm(name):
    assert isinstance(name, str)
    name = name.lower().replace("_")
    if name == "cifar10":
        return KDCifar10Norm()
    if name == "cifar100":
        return KDCifar100Norm()
    if name == "imagenet":
        return KDImageNetNorm()
    raise NotImplementedError
