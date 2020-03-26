from dlex.torch.utils.model_utils import RNN
from torch import nn
from torch.nn import LayerNorm
from torch.nn.modules import TransformerEncoderLayer, TransformerEncoder
from transformers import BertModel


class Encoder(nn.Module):
    def __init__(
            self,
            encoder_type,
            vocab_size,
            embed_dim,
            encoder_dim,
            output_dim,
            dropout,
            **kwargs):
        super().__init__()

        if encoder_type == "bert":
            self.embed = None
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            # for p in self.encoder.parameters():
            #     p.requires_grad = False
            # self.encoder.cuda()
        elif encoder_type == "transformer":
            self.embed = nn.Embedding(vocab_size, embed_dim)
            encoder_layer = TransformerEncoderLayer(encoder_dim, nhead=8)
            encoder_norm = LayerNorm(encoder_dim)
            self.encoder = TransformerEncoder(encoder_layer, 1, encoder_norm)
        elif encoder_type in ["lstm", "rnn", "gru"]:
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.encoder = RNN(
                encoder_type,
                embed_dim,
                output_dim,
                bidirectional=kwargs.get('bidirectional', False),
                dropout=dropout)

        if encoder_dim != output_dim or kwargs.get('project', False):
            self.encoder_output_linear = nn.Linear(encoder_dim, output_dim)
            self.encoder_state_linear = nn.Linear(encoder_dim, output_dim)
        else:
            self.encoder_output_linear = nn.Sequential()
            self.encoder_state_linear = nn.Sequential()

        self.encoder_type = encoder_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, input_lengths):
        if self.encoder_type == "bert":
            out = self.encoder(inputs)
            c = out[0]
            h = out[1]
        elif self.encoder_type == "transformer":
            embed = self.embed(inputs)
            c = self.encoder(embed.transpose(0, 1), src_key_padding_mask=~get_mask(input_lengths))
            c = c.transpose(0, 1)
            h = c[:, 0]
            # c = self.lstm_proj(c)
        else:
            embed = self.embed(inputs)
            embed = self.dropout(embed)
            c, h = self.encoder(embed, input_lengths)

        h = self.encoder_state_linear(h)
        c = self.encoder_output_linear(c)
        return c, h

    def reset(self):
        pass
        # if self.embed:
        #     self.embed.weight.data.uniform_(0, 1)