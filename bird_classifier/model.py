import torch.nn as nn
from torchvision.models import EfficientNet, efficientnet_b6, EfficientNet_B6_Weights

class EfficientNetModel(nn.Module):

    def __init__ (self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.feature_extractor = efficientnet_b6(weights = EfficientNet_B6_Weights.DEFAULT) 

        #freezes all the layers 
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        #only replace the last layer
        last_layer = self.feature_extractor.classifier[1]
        in_features = last_layer.in_features 
        self.feature_extractor.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
 
    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        return prediction
