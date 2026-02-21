# models/inception_time.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Keras-faithful Inception module (channels-first).

    Original Keras logic:
      - optional bottleneck conv1d (1x1) if channels > 1
      - 3 conv branches with kernel sizes: (kernel_size-1)//(2**i) for i=0..2
      - 1 maxpool branch + 1x1 conv
      - concat branches (Keras axis=2 for channels-last); here concat on dim=1
      - BN + ReLU
    """
    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        bottleneck_size: int = 32,
        kernel_size: int = 41,
        use_bottleneck: bool = True,
        stride: int = 1,
    ):
        super().__init__()

        self.use_bottleneck = bool(use_bottleneck and in_channels > 1)

        # Bottleneck
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_size, kernel_size=1, stride=1, padding=0, bias=False
            )
            inception_in = bottleneck_size
        else:
            self.bottleneck = None
            inception_in = in_channels

        # Keras sets: self.kernel_size = kernel_size - 1
        k = kernel_size - 1
        kernel_sizes = [max(3, k // (2 ** i)) for i in range(3)]

        # "same" padding-ish: prefer odd kernel sizes
        kernel_sizes = [ks if ks % 2 == 1 else ks + 1 for ks in kernel_sizes]

        self.conv_branches = nn.ModuleList()
        for ks in kernel_sizes:
            self.conv_branches.append(
                nn.Conv1d(
                    inception_in,
                    nb_filters,
                    kernel_size=ks,
                    stride=stride,
                    padding=ks // 2,
                    bias=False,
                )
            )

        # MaxPool branch + 1x1 conv
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.conv_pool = nn.Conv1d(
            in_channels, nb_filters, kernel_size=1, stride=1, padding=0, bias=False
        )

        # Output channels = 4 branches * nb_filters
        out_channels = nb_filters * 4
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        if self.bottleneck is not None:
            x_in = self.bottleneck(x)
        else:
            x_in = x

        outs = [conv(x_in) for conv in self.conv_branches]

        p = self.max_pool(x)
        p = self.conv_pool(p)
        outs.append(p)

        x = torch.cat(outs, dim=1)  # concat on channels
        x = self.bn(x)
        x = self.relu(x)
        return x


class ShortcutLayer(nn.Module):
    """
    Residual shortcut layer like Keras _shortcut_layer:
      shortcut_y = Conv1D(filters=out_channels, kernel_size=1)(input_tensor)
      BN
      Add + ReLU
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input_tensor: torch.Tensor, out_tensor: torch.Tensor) -> torch.Tensor:
        shortcut = self.bn(self.conv(input_tensor))
        x = self.relu(shortcut + out_tensor)
        return x


class InceptionTimeBase(nn.Module):
    """
    EEG-only InceptionTime, faithful to the Keras build_model loop:

      x = input
      input_res = input
      for d in range(depth):
          x = inception_module(x)
          if use_residual and d % 3 == 2:
              x = shortcut_layer(input_res, x)
              input_res = x
      gap -> (dropout) -> dense
    """
    def __init__(
        self,
        input_shape,
        nb_classes: int = 2,
        nb_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
        dropout: float = 0.3,   # NEW: dropout before FC
    ):
        super().__init__()

        self.n_chans, self.n_times = input_shape
        self.nb_classes = nb_classes
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size
        self.bottleneck_size = 32

        self.inception_modules = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        in_channels = self.n_chans
        residual_in_channels = in_channels  # channels at the start of each 3-layer block

        for d in range(depth):
            # At the start of every residual block (d=0,3,6...), remember the channels of input_res
            if self.use_residual and d % 3 == 0:
                residual_in_channels = in_channels

            self.inception_modules.append(
                InceptionModule(
                    in_channels=in_channels,
                    nb_filters=nb_filters,
                    bottleneck_size=self.bottleneck_size,
                    kernel_size=kernel_size,
                    use_bottleneck=use_bottleneck,
                    stride=1,
                )
            )

            out_channels = nb_filters * 4

            # Residual every 3 modules (d=2,5,8...)
            if self.use_residual and d % 3 == 2:
                self.shortcuts.append(ShortcutLayer(residual_in_channels, out_channels))

            in_channels = out_channels

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)  # NEW
        self.fc = nn.Linear(in_channels, nb_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns final feature map BEFORE GAP: (B, F, T)
        """
        input_res = x
        shortcut_idx = 0

        for d, inc in enumerate(self.inception_modules):
            x = inc(x)
            if self.use_residual and d % 3 == 2:
                x = self.shortcuts[shortcut_idx](input_res, x)
                input_res = x
                shortcut_idx += 1

        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.forward_features(x)           # (B, F, T)
        pooled = self.gap(feat).squeeze(-1)       # (B, F)
        pooled = self.dropout(pooled)             # NEW
        logits = self.fc(pooled)                  # (B, K)
        return logits


class InceptionTimeGazeInput(InceptionTimeBase):
    """
    Input-level gaze modulation:
      eeg <- eeg * (1 + alpha * gaze)
    """
    def __init__(self, input_shape, nb_classes=2, **kwargs):
        super().__init__(input_shape=input_shape, nb_classes=nb_classes, **kwargs)
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg: torch.Tensor, gaze: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if gaze is not None:
            # Expect gaze already (B,C,T). Runner will enforce/reshape if needed.
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        return super().forward(eeg)


class InceptionTimeGazeOutput(InceptionTimeBase):
    """
    Output-level attention map generation from final feature map.

    If return_attention=True:
      returns dict { 'logits': logits, 'attention_map': (B,C,T) }
    """
    def __init__(self, input_shape, nb_classes=2, **kwargs):
        super().__init__(input_shape=input_shape, nb_classes=nb_classes, **kwargs)

        feature_channels = self.nb_filters * 4
        self.attention_conv = nn.Conv1d(feature_channels, 1, kernel_size=3, padding=1)

        nn.init.xavier_uniform_(self.attention_conv.weight)
        nn.init.zeros_(self.attention_conv.bias)

    def forward(self, eeg: torch.Tensor, return_attention: bool = False, **kwargs):
        feat = self.forward_features(eeg)                 # (B, F, T)
        pooled = self.gap(feat).squeeze(-1)               # (B, F)
        pooled = self.dropout(pooled)                     # keep dropout consistent
        logits = self.fc(pooled)                          # (B, K)

        if not return_attention:
            return logits

        # attention at feature resolution -> upsample to original time
        att_low = torch.sigmoid(self.attention_conv(feat))  # (B,1,T)
        att_full = F.interpolate(att_low, size=self.n_times, mode="linear", align_corners=False)  # (B,1,T)
        att_full = att_full.repeat(1, self.n_chans, 1)      # (B,C,T)

        return {"logits": logits, "attention_map": att_full}


class InceptionTimeGazeBoth(InceptionTimeGazeOutput):
    """
    Both input modulation and output attention.
    """
    def __init__(self, input_shape, nb_classes=2, **kwargs):
        super().__init__(input_shape=input_shape, nb_classes=nb_classes, **kwargs)
        self.gaze_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        eeg: torch.Tensor,
        gaze: torch.Tensor = None,
        return_attention: bool = False,
        **kwargs
    ):
        if gaze is not None:
            eeg = eeg * (1.0 + self.gaze_alpha * gaze)
        return super().forward(eeg, return_attention=return_attention)