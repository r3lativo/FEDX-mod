"""
Main model (FedX) class representing backbone network and projection heads

"""

import torch.nn as nn

from resnetcifar import ResNet18_cifar10, ResNet50_cifar10


class ModelFedX(nn.Module):
    def __init__(self, base_model, out_dim, net_configs=None):
        super(ModelFedX, self).__init__()

        if (
            base_model == "resnet50-cifar10"
            or base_model == "resnet50-cifar100"
            or base_model == "resnet50-smallkernel"
            or base_model == "resnet50"
        ):
            basemodel = ResNet50_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            basemodel.fc.in_features
        elif base_model == "resnet18-fmnist":
            basemodel = ResNet18_mnist()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        elif base_model == "resnet18-cifar10" or base_model == "resnet18":
            basemodel = ResNet18_cifar10()
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            self.num_ftrs = basemodel.fc.in_features
        else:
            raise ("Invalid model type. Check the config file and pass one of: resnet18 or resnet50")

        self.projectionMLP = nn.Sequential(
            nn.Linear(self.num_ftrs, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

        self.predictionMLP = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)

        h.view(-1, self.num_ftrs)
        h = h.squeeze()

        proj = self.projectionMLP(h)
        pred = self.predictionMLP(proj)
        return h, proj, pred


def init_nets(net_configs, n_parties, args, device="cpu", method="default"):
    nets = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        if method == "simsiam":
            net = SimSiam(args.model, args.out_dim, net_configs)
        else:
            net = ModelFedX(args.model, args.out_dim, net_configs)
        net = net.cuda()
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type



# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# ADAPTED

class SimSiam(ModelFedX):
    def __init__(self, base_model, output_dim, net_configs=None):
        super(ModelFedX, self).__init__()

        if base_model == "resnet18":
            basemodel = ResNet18_cifar10()
        elif base_model == "resnet50":
            basemodel = ResNet50_cifar10()
        else:
            raise ("Invalid model type. Check the config file and pass one of: resnet18 or resnet50")
        
        self.encoder = nn.Sequential(*list(basemodel.children())[:-1])
        self.num_ftrs = basemodel.fc.in_features

        proj_hidden_dim: int = self.num_ftrs
        proj_output_dim: int = output_dim
        pred_hidden_dim: int = output_dim


        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.num_ftrs, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(start_dim=1)
        z = self.projector(h)
        p = self.predictor(z)
        return z, p