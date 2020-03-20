from typing import Tuple, List

import h5py
import torch
import numpy as np
from dlex import logger
from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.variable_length_tensor import pad_sequence


class PytorchCLEVR(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

        self._vocab = None
        self._answers = None
        self.h = h5py.File(builder.get_image_features_path(self.mode, "mac"), 'r')
        self.image_features = self.h['features']

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = Vocab.from_file(self.builder.vocab_path)
        return self._vocab

    @property
    def answers(self):
        if self._answers is None:
            self._answers = open(self.builder.answer_path, "r").read().strip().split('\n')
        return self._answers

    @property
    def num_classes(self):
        return len(self.answers)

    @property
    def data(self):
        if self._data is None:
            self._data = []
            with open(self.builder.get_data_path(self.mode)) as f:
                lines = [l for l in f.read().split('\n') if l.strip() != ""]
                logger.info(f"Dataset loaded. Number of samples: {len(lines):,}")

            for line in lines:
                img_path, q, _, a = line.split('\t')
                self._data.append([img_path, self.vocab.encode_token_list(q.split(' ')), int(a)])

            # self._data.sort(key=lambda d: len(d[1]))
            logger.info(
                "Question length - max: %d - avg: %d",
                max(len(d[1]) for d in self._data),
                np.average([len(d[1]) for d in self._data]))
        return self._data

    def close(self):
        self.h.close()

    def __getitem__(self, i):
        if type(i) == int:
            _i = slice(i, i + 1)
        else:
            _i = i
        img_path, q, ans = zip(*self.data[_i])
        img = [torch.from_numpy(self.image_features[int(path.rsplit('_', 1)[1][:-4])]) for path in img_path]
        return list(zip(img, q, ans))[0] if type(i) == int else list(zip(img, q, ans))

    def collate_fn(self, batch: List[Tuple]):
        batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
        imgs, qs, ans = [[b[i] for b in batch] for i in range(3)]
        qs, qlen = pad_sequence(qs, self.vocab.blank_token_idx, True)

        return Batch(
            X=(self.maybe_cuda(torch.stack(imgs)), qs, qlen),
            Y=self.maybe_cuda(torch.LongTensor(ans)))

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        return " ".join(self.vocab.decode_idx_list(batch_input.X[1][:batch_input.X[2]])), \
               self.answers[y_pred], \
               self.answers[batch_input.Y]