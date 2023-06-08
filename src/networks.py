import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, vgg16
from torchvision.ops import roi_pool

class WSDDN(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = vgg16(pretrained=True)
        self.roi_output_size = (7, 7)

        self.features = self.base.features[:-1]

        self.fcs = self.base.classifier[:-1]
        self.fc_c = nn.Linear(4096, 512)
        self.fc_d = nn.Linear(4096, 512)

    def forward(self, batch_imgs, batch_boxes, query_feat):

        feat = self.features(batch_imgs)  # [B, 512, H, W]

        scale = feat.shape[-1] / batch_imgs.shape[-1]
        all_combined_scores = []
        for idx, boxes in enumerate(batch_boxes):
            out = roi_pool(feat[idx].unsqueeze(0), [boxes], self.roi_output_size, scale)
            out = out.view(len(boxes), -1) # [N, 512]

            out = self.fcs(out)  # [4000, 4096]

            classification_scores = F.softmax(F.cosine_similarity(self.fc_c(out).unsqueeze(1), query_feat[idx].unsqueeze(0), dim=-1), dim=1)
            detection_scores = F.softmax(F.cosine_similarity(self.fc_d(out).unsqueeze(1), query_feat[idx].unsqueeze(0), dim=-1), dim=0)
            combined_scores = classification_scores * detection_scores
            all_combined_scores.append(combined_scores)

        return all_combined_scores

    @staticmethod
    def calculate_loss(all_combined_scores, target):
        all_image_level_scores = []
        for combined_scores in all_combined_scores:
            image_level_scores = torch.sum(combined_scores, dim=0)
            image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
            all_image_level_scores.append(image_level_scores)
        all_image_level_scores = torch.stack(all_image_level_scores) # B x C        

        # loss = F.binary_cross_entropy(all_image_level_scores, target, reduction="sum")
        loss = F.cross_entropy(all_image_level_scores, torch.argmax(target, dim=-1), reduction="mean")
        return loss
