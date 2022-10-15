from json import encoder
import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # (b, 3, h, w) -> (b, 2048, h/32, w/32)
        out = self.resnet(images)
        # -> (b, 2048, self.enc_iamge_size, self.enc_image_size)
        out = self.adaptive_pool(out)
        # -> (b, self.enc_image_size, self.enc_image_size, 2048)
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
        # layer n encoded -> (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)
        # layer n-1 hidden -> (batch_size, attention_dim)
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
        # it is a lookup tabel(查找表)
        # it will init a tabel:weight (vocab_size, embed_dim) each word index i has weight[i] (embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = None if self.dropout <= 0 else nn.Dropout(p=self.dropout)
        # decoding LSTMCell
        # LSTMCell is just computation cell for one layer and one word
        # it need compose and cycle to build layers and sequence
        # input dim = embed_dim + encoder_dim because we need embed word and encoded image
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        # from lstm output to word 
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
        :return hidden state, cell state (batch_size, decoder_dim)
        """
        # init has no weighted, so use mean
        # -> (batch_size, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1)
        # -> (batch_size ,decoder_dim)
        h = self.init_h(mean_encoder_out)
        # -> (batch_size ,decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation

        :param encoder_out: encoded images (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions (batch_size, max_caption_length)
        :param caption_lengths: caption lengths (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # flatten image 
        # (batch_size, encoded_image_size, encoded_image_size, encoder_dim) -> (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # sort inut data by decreasing lengths
        # sort return: sorted_value, sorted_index
        # descending=True 降序排序 从大到小 descending=False 升序排序 从小到大
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # (batch_size, max_caption_lengths) -> (batch_size, max_caption_lengths, embedding_dim)
        # embed captions: each word i map embedding.weight[i] (embed_dim)
        embeddings = self.embedding(encoded_captions)

        # get init hidden state and cell state (batch_size, decoder_dim)
        h,c = self.init_hidden_state(encoder_out)

        # do not decode <end>, so need caption_length - 1 steps 
        decode_lengths = (caption_lengths - 1).tolist()

        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            # decode_lengths is descending order, so [:batch_size_t]means need decode images
            batch_size_t = sum([l > t for l in decode_lengths])
            # attention_weighted_encoding: (batch_size_t, encoder_dim)
            # alpha: (batch_size_t, num_pixels)
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            # gate (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            # embeddings[:batch_size_t, t, :]: (batch_size_t, embed_dim)
            # attention_weighted_encoding[:batch_size_t, :]: (batch_size_t, encoder_dim)
            # torch.cat(up two): (batch_size, embed_dim, encoder_dim) 
            h,c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h) if self.dropout is not None else h)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
    
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
