import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:   
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = l(h)
            h = nn.ReLU()(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], dim=-1)

            for i, l in enumerate(self.views_linears):
                h = l(h)
                h = nn.ReLU()(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

# Create a sample input tensor for NeRF
batch_size = 4
input_ch = 3
input_ch_views = 3
sample_input = torch.randn(batch_size, input_ch + input_ch_views)

# Initialize the NeRF model
nerf_model = NeRF(use_viewdirs=True)

# Forward pass
outputs = nerf_model(sample_input)

print(outputs.shape)