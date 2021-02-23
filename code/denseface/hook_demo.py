import torch
import torch.nn as nn
from torch.nn import init

class MidLayerFeatureExtractor(object):
    def __init__(self, layer):
        self.layer = layer
        self.feature = None
        self.layer.register_forward_hook(self.hook)
        self.device = None
    
    def hook(self, module, input, output):
        # default tensor on cpu
        self.is_empty = True
        self.feature = output.clone()
        self.is_empty = False
    
    def extract(self):
        assert not getattr(self, "is_empty"), 'Synic Error in MidLayerFeatureExtractor, \
                this may caused by calling extract method before the hooked module has execute forward method'
        return self.feature

class MultiLayerFeatureExtractor(object):
    def __init__(self, net, layers):
        '''
        Parameter:
        -----------------
        net: torch.nn.Modules
        layers: str, something like "C.fc[0], module[1]"
                which will get mid layer features in net.C.fc[0] and net.module[1] respectively
        '''
        self.net = net
        self.layer_names = layers
        self.layers = [self.str2layer(layer_name) for layer_name in self.layer_names]
        self.extractors = [MidLayerFeatureExtractor(layer) for layer in self.layers]

    def str2layer(self, name):
        modules = name.split('.')
        layer = self.net
        for module in modules:
            if '[' and ']' in module:
                sequential_name = module[:module.find('[')]
                target_module_num = int(module[module.find('[')+1:module.find(']')])
                layer = getattr(layer, sequential_name)
                layer = layer[target_module_num]
            else:
                layer = getattr(layer, module)
        return layer
    
    def extract(self):
        ans = [extractor.extract() for extractor in self.extractors]
        return ans

if __name__ == '__main__':
    netA = nn.Sequential(
        nn.Linear(20, 30),
        nn.Linear(30, 40),
        nn.ReLU(),
        nn.Linear(40, 50)
    )
    netB = nn.Sequential(
        nn.Linear(50, 60),
        nn.Linear(60, 70),
        nn.Linear(70, 80)
    )

    class ExampleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.A = netA
            self.B = netB
        
        def forward(self, x):
            return self.B(self.A(x))
    
    example = ExampleNet()
    print(example)
    # 如果要抽取netA的第一层和netB的第二层:
    ex = MultiLayerFeatureExtractor(example, ['A[0]', 'B[0]'])
    x_in = torch.rand(1, 20)
    x_out = example(x_in)
    a_1, b_1 = ex.extract()
    print(a_1.shape)
    print(b_1.shape)
