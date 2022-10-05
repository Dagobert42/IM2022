from torch import nn
from experiments.form_gan.utils import *
from experiments.form_gan.small_resnet import *

class SegmentationTransformer(nn.Module):
    def __init__(self, transform_dim, num_classes, num_queries):
        super(SegmentationTransformer, self).__init__()
        transform_dim = transform_dim
        self.num_queries = num_queries

        self.backbone = ResNet([1, 1, 1, 1], num_classes)
        self.convert = nn.Conv3d(512, transform_dim, kernel_size=1)
        self.pos_encoding = SineEncoding(transform_dim // 2, normalize=True)
        self.transformer = nn.Transformer(transform_dim)

        self.class_embed = nn.Linear(transform_dim, num_classes + 1)
        self.bbox_embed = MLP(transform_dim, transform_dim, 4, 3)
        self.real_fake_embed = nn.Linear(transform_dim, 1)
        self.query_embed = nn.Embedding(num_queries, transform_dim)
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.convert(out)

        pos = self.pos_encoding(x)
        out = self.transformer(
            pos + 0.1 * out.flatten(2).permute(2, 0, 1),
            self.query_pos.unsqueeze(1)
            ).transpose(0, 1)
        
        classes = self.class_embed(out)
        bboxes = self.bbox_embed(out).sigmoid()
        score = self.real_fake_embed(out)
        return classes[-1], bboxes, score