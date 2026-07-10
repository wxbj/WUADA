import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet3PlusDecoderBlock(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels, target_index, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.target_index = target_index

        self.enc_compress = nn.ModuleList([
            ConvBNReLU(c, out_channels, norm_layer) for c in encoder_channels
        ])

        self.dec_compress = nn.ModuleList([
            ConvBNReLU(c, out_channels, norm_layer) for c in decoder_channels
        ])

        total_inputs = len(encoder_channels) + len(decoder_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * total_inputs, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, encoder_feats, decoder_feats):
        target_size = encoder_feats[self.target_index].shape[2:]
        out_feats = []

        for feat, conv in zip(encoder_feats, self.enc_compress):
            x = conv(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            out_feats.append(x)

        for feat, conv in zip(decoder_feats, self.dec_compress):
            x = conv(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
            out_feats.append(x)

        return self.fuse(torch.cat(out_feats, dim=1))


class UNet3PlusDecoder(nn.Module):
    def __init__(self, num_classes, encoder_channels, out_channels=64, deep_supervision=True):
        super().__init__()
        self.stage = len(encoder_channels)
        self.deep_supervision = deep_supervision
        self.out_channels = out_channels
        self.norm_layer=nn.BatchNorm2d

        self.decoders = nn.ModuleList()
        for i in range(self.stage):
            dec_input_channels = [out_channels] * (self.stage - i - 1)
            block = UNet3PlusDecoderBlock(
                encoder_channels=encoder_channels,
                decoder_channels=dec_input_channels,
                out_channels=out_channels,
                target_index=i,
                norm_layer=self.norm_layer
            )
            self.decoders.append(block)

        self.final = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_heads = nn.ModuleList([
                nn.Conv2d(out_channels, num_classes, kernel_size=1) for _ in range(self.stage)
            ])

    def forward(self, features, size=None):
        feat_list = [features[f"x{i}"] for i in range(self.stage)]
        decoder_outputs = [None] * self.stage

        for i in reversed(range(self.stage)):
            deeper_outputs = decoder_outputs[i + 1:] if i < self.stage - 1 else []
            decoder_outputs[i] = self.decoders[i](feat_list, deeper_outputs)

        out = self.final(decoder_outputs[0])
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            aux_outs = []
            for i, head in enumerate(self.ds_heads):
                aux = head(decoder_outputs[i])
                if size is not None:
                    aux = F.interpolate(aux, size=size, mode='bilinear', align_corners=True)
                aux_outs.append(aux)
            return [out] + aux_outs

        return out

