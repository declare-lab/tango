import torch
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, Inception3
from collections import OrderedDict


def load_module2model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
        if k[:7] == "module.":
            name = k[7:]  # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
    return new_state_dict


class Melception(Inception3):
    def __init__(
        self, num_classes, features_list, feature_extractor_weights_path, **kwargs
    ):
        # inception = Melception(num_classes=309)
        super().__init__(num_classes=num_classes, init_weights=True, **kwargs)
        self.features_list = list(features_list)
        # the same as https://github.com/pytorch/vision/blob/5339e63148/torchvision/models/inception.py#L95
        # but for 1-channel input instead of RGB.
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        # also the 'hight' of the mel spec is 80 (vs 299 in RGB) we remove all max pool from Inception
        self.maxpool1 = torch.nn.Identity()
        self.maxpool2 = torch.nn.Identity()

        state_dict = torch.load(feature_extractor_weights_path, map_location="cpu")
        new_state_dict = load_module2model(state_dict["model"])
        # print('before....')
        # print(self.state_dict()['Conv2d_1a_3x3.conv.weight'])
        # print('after')
        self.load_state_dict(new_state_dict)
        # print(self.state_dict()['Conv2d_1a_3x3.conv.weight'])
        # assert 1==2
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        features = {}
        remaining_features = self.features_list.copy()

        # B x 1 x 80 x 848 <- N x M x T
        x = x.unsqueeze(1)
        # (B, 32, 39, 423) <-
        x = self.Conv2d_1a_3x3(x)
        # (B, 32, 37, 421) <-
        x = self.Conv2d_2a_3x3(x)
        # (B, 64, 37, 421) <-
        x = self.Conv2d_2b_3x3(x)
        # (B, 64, 37, 421) <-
        x = self.maxpool1(x)

        if "64" in remaining_features:
            features["64"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("64")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 80, 37, 421) <-
        x = self.Conv2d_3b_1x1(x)
        # (B, 192, 35, 419) <-
        x = self.Conv2d_4a_3x3(x)
        # (B, 192, 35, 419) <-
        x = self.maxpool2(x)

        if "192" in remaining_features:
            features["192"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("192")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 256, 35, 419) <-
        x = self.Mixed_5b(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_5c(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_5d(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_6a(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6b(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6c(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6d(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6e(x)

        if "768" in remaining_features:
            features["768"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("768")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 1280, 8, 104) <-
        x = self.Mixed_7a(x)
        # (B, 2048, 8, 104) <-
        x = self.Mixed_7b(x)
        # (B, 2048, 8, 104) <-
        x = self.Mixed_7c(x)
        # (B, 2048, 1, 1) <-
        x = self.avgpool(x)
        # (B, 2048, 1, 1) <-
        x = self.dropout(x)

        # (B, 2048) <-
        x = torch.flatten(x, 1)
        # print('x ',x.shape)
        if "2048" in remaining_features:
            features["2048"] = x
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        if "logits_unbiased" in remaining_features:
            # (B, num_classes) <-
            x = x.mm(self.fc.weight.T)
            features["logits_unbiased"] = x
            remaining_features.remove("logits_unbiased")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        features["logits"] = x
        # print('x ',x.shape)
        # assert 1==2
        return tuple(features[a] for a in self.features_list)

    def convert_features_tuple_to_dict(self, features):
        """
        The only compound return type of the forward function amenable to JIT tracing is tuple.
        This function simply helps to recover the mapping.
        """
        message = "Features must be the output of forward function"
        assert type(features) is tuple and len(features) == len(
            self.features_list
        ), message
        return dict(
            ((name, feature) for name, feature in zip(self.features_list, features))
        )
