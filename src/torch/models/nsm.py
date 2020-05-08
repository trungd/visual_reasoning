import torch
import torch.nn as nn
import torch.nn.functional as F

from dlex import Params
from dlex.datasets.torch import Dataset
from dlex.torch.models import BaseModel
from dlex.torch.utils.model_utils import RNN

from ...datatypes import BatchX


class NSM(BaseModel):
    def __init__(self, params: Params, dataset: Dataset):
        super().__init__(params, dataset)

        cfg = self.configs
        self.W = torch.eye(cfg.embed_dim, requires_grad=True)
        self.property_W = torch.stack([torch.eye(cfg.embed_dim, requires_grad=True) for _ in range(self.num_attributes + 1)], dim=0)
        self.W_L_plus_1 = torch.eye(cfg.embed_dim, requires_grad=True)
        self.W_r = nn.Linear(cfg.embed_dim, 1, bias=False)
        self.W_s = nn.Linear(cfg.embed_dim, 1, bias=False)

        self.encoder = RNN(cfg.encoder.type, cfg.embed_dim, cfg.embed_dim, bidirectional=False, dropout=0.)
        self.decoder = RNN(cfg.decoder.type, cfg.embed_dim, cfg.embed_dim, bidirectional=False, dropout=0.)
        self.classifier = nn.Sequential(
            nn.Linear(2 * cfg.embed_dim, 2 * cfg.embed_dim),
            nn.ELU(),
            nn.Linear(2 * cfg.embed_dim, dataset.num_classes))

        self.concept_emb = nn.Embedding()  # C + 1
        # add non-structural concept embedding
        self.concept_emb = torch.cat([self.concept_emb, torch.rand(1, cfg.embed_dim, requires_grad=True)])

        self.attribute_emb = nn.Embedding()  # L + 2

    @property
    def num_nodes(self):
        return self.configs.num_nodes

    @property
    def num_attributes(self):
        return 3

    def next_state(self, p: torch.Tensor, r: torch.Tensor, states, edges):
        batch_size = p.shape[0]
        embed_dim = self.configs.embed_dim

        # instruction type
        R = torch.bmm(self.attribute_emb.expand(batch_size, -1, embed_dim), r.unsqueeze(2))
        R = F.softmax(R, 1).squeeze(2)

        # r_i_prime is "degree to which that reasoning instruction is concerned with semantic relations"
        r_prime = R[:, -1].unsqueeze(1)
        property_R_i = R[:, :-1]

        # bilinear proyecctions (one for each property) initialized to identity.
        gamma_s = torch.sum(
            torch.mul(
                property_R_i.view(batch_size, -1, 1, 1),
                torch.mul(
                    torch.matmul(
                        states.transpose(2, 1),
                        self.property_W
                    ), r.view(batch_size, 1, 1, embed_dim)
                )
            ), dim=1)
        gamma_s = F.elu(gamma_s)

        # bilinear proyecction
        gamma_e = torch.mul(
            torch.bmm(
                edges.view(batch_size, -1, embed_dim),
                self.W_L_plus_1.expand(batch_size, embed_dim, embed_dim)
            ), r.unsqueeze(1))
        gamma_e = F.elu(gamma_e)
        gamma_e = gamma_e.view(batch_size, self.num_nodes, self.num_nodes, embed_dim)

        p_r = self.W_r(torch.sum(torch.mul(gamma_e, p.view(batch_size, -1, 1, 1)), dim=1))
        p_r = F.softmax(p_r.squeeze(2), 1)

        p_s = F.softmax(self.W_s(gamma_s).squeeze(2), dim=1)

        return r_prime * p_r + (1 - r_prime) * p_s

    def forward(self, batch):
        cfg = self.configs
        bx = BatchX(*self.to_cuda_tensors(batch.X))
        batch_size = bx.questions.shape[0]

        states = None
        edges = None

        word_concept_similarity = torch.bmm(
            torch.bmm(
                bx.questions,
                self.W.expand(batch_size, cfg.embed_dim, cfg.embed_dim)
            ),
            self.concept_emb.expand(batch_size, -1, cfg.embed_dim).transpose(1, 2)
        )
        word_concept_similarity = torch.softmax(word_concept_similarity, -1)

        # concept-based representation of each word
        V = (word_concept_similarity[:, :, -1]).unsqueeze(2) * bx.questions + \
            torch.bmm(word_concept_similarity[:, :, :-1], C[:-1, :].expand(batch_size, -1, cfg.embed_dim))

        _, encoder_hidden = self.encoder(V)
        q, _ = encoder_hidden
        q = q.view(batch_size, 1, cfg.embed_dim)

        h, _ = self.decoder(q.expand(batch_size, cfg.num_steps + 1, cfg.embed_dim), encoder_hidden)
        instructions = torch.bmm(torch.softmax(torch.bmm(h, V.transpose(1, 2)), dim=2), V)

        p = torch.ones(batch_size, self.num_nodes, self.num_nodes)

        for i in range(cfg.num_steps):
            p = self.next_state(p, instructions[:, i, :], states, edges)

        # Classify
        r = instructions[:, -1, :]
        property_R = F.softmax(
            torch.bmm(
                self.attribute_emb.expand(batch_size, -1, cfg.embed_dim),
                r.unsqueeze(2)
            ), dim=1).squeeze(2)[:, :-1]

        # equivalent to:torch.sum(p_i.unsqueeze(2) * torch.sum(property_R_N.view(10, 1, 3, 1) * S, dim=2), dim=1)
        m = torch.bmm(
            p.unsqueeze(1),
            torch.sum(property_R.view(batch_size, 1, self.num_attributes + 1, 1) * S, dim=2)
        )

        pre_logits = self.classifier(torch.cat([m, q], dim=2).squeeze(1))

        return pre_logits