from collections import namedtuple

import torch
import torch.nn.functional as F
from dlex import Params
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.models import ClassificationModel
from dlex.torch.utils.model_utils import MultiLinear
from .encoder import Encoder

from ...datatypes import BatchX
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super().__init__()

        self.position_aware = nn.ModuleList(linear(dim, dim) for _ in range(max_step))
        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, c_prev):
        position_aware = self.position_aware[step](question)

        # 1. Calculate cq_i
        cq = torch.cat([c_prev, position_aware], 1)  # control_question
        cq = self.control_question(cq)
        cq = cq.unsqueeze(1)

        # 2. Attention distribution over the question words
        context_prod = cq * context
        attn_weight = self.attn(context_prod)
        attn = F.softmax(attn_weight, 1)
        c = (attn * context).sum(1)
        return c


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memories, k, c):
        """
        :param memories:
        :param k: knowledge
        :param c: control
        :return: r_i
        """
        # 1. Interaction between knowledge k_{h,w} and memory m_{i-1}
        m_prev = memories[-1]
        I = self.mem(m_prev).unsqueeze(2) * k  # I_{i,h,w}

        # 2. Calculate I'_{i,h,w}
        I = self.concat(torch.cat([I, k], 1).permute(0, 2, 1))

        # 3. Attention distribution over knowledge base elements
        attn = I * c[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)
        read = (attn * k).sum(2)
        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super().__init__()

        self.concat = linear(dim * 2, dim)
        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)
        if memory_gate:
            self.control = linear(dim, 1)
        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, r, controls):
        """
        :param memories:
        :param r: retrieved
        :param controls:
        :return: m_i
        """
        m_prev = memories[-1]
        m = self.concat(torch.cat([r, m_prev], 1))

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            m = self.mem(attn_mem) + m

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = torch.sigmoid(control)
            m = gate * m_prev + (1 - gate) * m

        return m


class MACUnit(nn.Module):
    def __init__(self, dim, max_step, self_attention: bool, memory_gate: bool, dropout):
        super().__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)
        return mask

    def forward(self, context, question, knowledge):
        """
        :param context: batch_size x max_len x dim
        :param question: batch_size x (2 x dim)
        :param knowledge:
        :return:
        """
        batch_size = question.size(0)
        control = self.control_0.expand(batch_size, self.dim)
        memory = self.mem_0.expand(batch_size, self.dim)
        controls, memories = [control], [memory]

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask
            for i in range(self.max_step):
                control = self.control(i, context, question, control)
                control = control * control_mask
                controls.append(control)

                read = self.read(memories, knowledge, controls)  # batch_size x dim
                memory = self.write(memories, read, controls)  # batch_size x dim
                memory = memory * memory_mask
                memories.append(memory)
        else:
            for i in range(self.max_step):
                control = self.control(i, context, question, control)
                controls.append(control)

                read = self.read(memories, knowledge, controls)
                memory = self.write(memories, read, controls)
                memories.append(memory)

        return memory


class MAC(ClassificationModel):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)

        cfg = self.configs

        # Input Unit
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.image.num_channels, cfg.dim, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(cfg.dim, cfg.dim, 3, padding=1),
            nn.ELU())

        if cfg.encoder.type == "input":
            self.encoder = None
            self.encoder_output_linear = nn.Linear(cfg.encoder.dim, cfg.dim)
            self.encoder_state_linear = nn.Linear(cfg.encoder.dim, cfg.dim)
        else:
            self.encoder = Encoder(
                cfg.encoder.type,
                vocab_size=dataset.vocab_size,
                embed_dim=cfg.embed_dim,
                encoder_dim=cfg.encoder.dim,
                output_dim=cfg.dim,
                dropout=cfg.dropout,
                bidirectional=cfg.encoder.bidirectional,
                project=cfg.encoder.project)

        # MAC Unit
        self.mac = MACUnit(
            cfg.dim,
            cfg.max_step,
            cfg.self_attention,
            cfg.memory_gate,
            cfg.dropout)

        # Output Unit
        output_dim = cfg.dim + (cfg.dim if cfg.classifier.use_question else 0)
        self.linear = MultiLinear(
            [output_dim] + cfg.classifier.dims + [dataset.num_classes],
            dropout=cfg.dropout,
            norm_layer=None,
            activation_fn='elu')

        self.dropout = nn.Dropout(cfg.dropout)

        self.reset()

        # Metrics
        self._accuracy = 0

    def reset(self):
        if self.encoder:
            self.encoder.reset()
        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()
        kaiming_uniform_(self.linear[0].weight)

    def encode_image(self, image):
        img = self.conv(image)
        img = img.view(img.shape[0], self.configs.dim, -1)
        return img

    def forward(self, batch: Batch):
        bx = BatchX(*self.to_cuda_tensors(batch.X))

        # Input Unit
        img = self.encode_image(bx.images)
        if self.encoder:
            c, h = self.encoder(bx.questions, bx.question_lengths)
        else:
            c = bx.question_bert_outputs
            h = bx.question_bert_states
            h = self.encoder_state_linear(h)
            c = self.encoder_output_linear(c)

        # MAC Cell
        final_memory = self.mac(c, h, img)

        # Output Unit
        if self.configs.classifier.use_question:
            out = torch.cat([final_memory, h], 1)
        else:
            out = final_memory

        out = self.linear(out)
        return out

    def reset_counter(self):
        super().reset_counter()
        self._accuracy = 0