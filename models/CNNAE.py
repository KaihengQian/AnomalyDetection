import torch.nn as nn


# Encoder Class
class Encoder(nn.Module):
    def __init__(self, dropout):
        super(Encoder, self).__init__()
        self.cnn_enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.cnn_enc(x)
        return out


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, dropout, use_act):
        super(Decoder, self).__init__()
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()

        self.cnn_dec = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, z):
        dec_out = self.cnn_dec(z)
        if self.use_act:
            dec_out = self.act(dec_out)

        return dec_out


# 2DCNN Auto-Encoder Class
class CNNAE(nn.Module):
    def __init__(self, dropout_ratio, use_act=True):
        super(CNNAE, self).__init__()

        self.dropout_ratio = dropout_ratio

        self.encoder = Encoder(dropout=dropout_ratio)
        self.decoder = Decoder(dropout=dropout_ratio, use_act=use_act)

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)

        return x_dec

