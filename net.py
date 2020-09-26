from easydl import *
from torchvision import models
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet34, resnet101, resnet152, densenet121, densenet161, resnext101_32x8d, wide_resnet101_2, resnext50_32x4d, wide_resnet50_2
from config import *
from data import *

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.__in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features



class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.__in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features

class ResNet152Fc(nn.Module):
    def __init__(self):
        super(ResNet152Fc, self).__init__()
        self.resnet = resnet152(pretrained=True)
        self.__in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.__in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features

class DenseNet121Fc(nn.Module):
    def __init__(self):
        super(DenseNet121Fc, self).__init__()
        self.resnet = densenet121(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features

class DenseNet161Fc(nn.Module):
    def __init__(self):
        super(DenseNet161Fc, self).__init__()
        self.resnet = densenet169(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features


class ResNext101Fc(nn.Module):
    def __init__(self):
        super(ResNext101Fc, self).__init__()
        self.resnet = resnext_101_32x8d(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        # res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features

class ResNext50Fc(nn.Module):
    def __init__(self):
        super(ResNext50Fc, self).__init__()
        self.resnet = resnext50_32x4d(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        # res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features


class WideResNet50Fc(nn.Module):
    def __init__(self):
        super(WideResNet50Fc, self).__init__()
        self.resnet = wide_resnet50_2(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        # res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features


class WideResNet101Fc(nn.Module):
    def __init__(self):
        super(WideResNet101Fc, self).__init__()
        self.resnet = wide_resnet101_2(pretrained=True)
        self.__in_features = self.resnet.classifier.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        # res = F.avg_pool2d(res, 7)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features



class ResNet101Fc(nn.Module):
    def __init__(self):
        super(ResNet101Fc, self).__init__()
        self.resnet = resnet101(pretrained=True)
        self.__in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
    def forward(self, x):
        res = self.resnet(x)
        res = res.view(res.size(0), -1)
        return res
    def output_num(self):
        return self.__in_features



class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features
     
class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out



class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x_ = self.grl(x)
        y = self.main(x_)
        return y





model_dict = {
    'resnet34': ResNet34Fc,
    'resnet18': ResNet18Fc,
    'resnet50': ResNet50Fc,
    'resnet101': ResNet101Fc,
    'resnet152': ResNet152Fc,
    'densenet121': DenseNet121Fc,
    'densenet161': DenseNet161Fc,
    'wideresnet50':WideResNet50Fc,
    'wideresnet101': WideResNet101Fc,
    'resnext101': ResNext101Fc,
    'resnext50': ResNext50Fc,
    
    #  'resnext': ResNext101Fc,
    'vgg16': VGG16Fc
}



class TotalNet(nn.Module):

    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model]()
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.discriminator_separate = AdversarialNetwork(256)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0
