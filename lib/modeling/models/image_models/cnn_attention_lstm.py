import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, enc_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = enc_image_size
        
        resnet = torchvision.models.resnet101(pretrained=True)
        # remove pool and linear because we do not do classification
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
        # just fine_tune conv2 to conv4 
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
        :param encoder_out: encoded images (batch_size num_pixels encoder_dim) 
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
    
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size fo decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = None if self.dropout <= 0 else nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        # initial some layers with values from the uniform distribution
        self.init_weights()

    def init_weights(self):
        """
        Initialize some parametres with values from the uniform distribution
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings
        
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
    
    def fine_tune_embeddings(self, fine_tune=True):
        """
        Set allow fine-tune of embedding layer?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Create the initial hidden and cell states for the decoder's LSTM based on the encoded image.

        :param encoder_out: encoded images (batch_size, num_pixels, encoder_dim)
        :return hidden state, cell state
        """
        # -> batch_size encoder_dim
        mean_encoder_out = encoder_out.mean(dim=1)
        # -> b decoder_dim
        h = self.init_h(mean_encoder_out)
        # -> b decoder_dim
        c = self.init_c(mean_encoder_out)
        return h, c

    