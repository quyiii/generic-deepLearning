import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.enc_image_size = 14
        
        resnet = torchvision.models.resnet101(pretrained=True)
        # remove linear and pool because we do not do classification
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # resize image to given size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_image_size, self.enc_image_size))
        self.fine_tune()

    def forward(self, images):
        # b 3 h w -> b 2048 h/32 w/32
        out = self.resnet(images)
        # -> b 2048 self.enc_iamge_size self.enc_image_size
        out = self.adaptive_pool(out)
        # -> b self.enc_image_size self.enc_image_size 2048
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        :param encoder_out: encoder images (batch_size num_pixels encoder_dim) 
        :param decoder_hidden: previous decoder output (batch_size decoder_dim)
        :return attention weighted encoding, weights
        """
        # -> (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # -> (batch_size, attention_dim)
        att2 = self.decoder_att(decoder_hidden)
        # -> (batch_size, num_pixels)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        # -> (batch_size, encoder_dim)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha