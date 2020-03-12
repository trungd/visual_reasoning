from collections import namedtuple
from typing import Dict, Any

import h5py
import tensorflow.compat.v1 as tf
import numpy as np
from dlex import logger, List
from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.tf import Dataset
from dlex.tf.utils.utils import pad_sequence

Sample = namedtuple("Sample", "image_path question answer")
Placeholders = namedtuple("Placeholders", "images questions question_lengths answers")


class TensorflowGQA(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

        self._vocab = None
        self._answers = None
        self.h = h5py.File(builder.get_image_features_path(self.mode, "mac"), 'r')
        self.image_features = self.h['features']
        # self.image_idx = self.h['indices']
        self._word_embeddings = None

        self._placeholders = None

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
                self._data.append(Sample(
                    image_path=img_path,
                    question=self.vocab.encode_token_list(q.split(' ')),
                    answer=int(a)))

            # self._data.sort(key=lambda d: len(d[1]))
            logger.info(
                "Question length - max: %d - avg: %d",
                max(len(d[1]) for d in self._data),
                np.average([len(d[1]) for d in self._data]))
        return self._data

    def close(self):
        self.h.close()

    def get_sliced_batch(self, placeholders, start, end) -> Placeholders:
        return Placeholders(
            questions=placeholders.questions[start:end],
            question_lengths=placeholders.question_lengths[start:end],
            images=placeholders.images[start:end],
            answers=placeholders.answers[start:end])

    def populate_feed_dict(
            self,
            feed_dict: Dict[tf.placeholder, Any],
            placeholders: Placeholders,
            data: List[Sample]) -> None:
        qs, qlen = pad_sequence([s.question for s in data])
        feed_dict[placeholders.questions] = qs
        feed_dict[placeholders.question_lengths] = qlen
        feed_dict[placeholders.images] = [self.image_features[int(s.image_path.rsplit('_', 1)[1][:-4])] for s in data]
        feed_dict[placeholders.answers] = [s.answer for s in data]

    def get_word_embeddings(self, size, dim, initializing='uniform', scale=1.):
        if initializing == 'normal':
            return np.random.randn(size, dim)
        elif initializing == 'uniform':
            return np.random.uniform(
                low=-scale,
                high=scale,
                size=(size, dim))

    @property
    def word_embeddings(self):
        if not self._word_embeddings:
            return dict(
                q=self.get_word_embeddings(self.vocab_size, self.configs.embedding.dim),
                a=self.get_word_embeddings(len(self.answers), self.configs.embedding.dim)
            )
        return self._word_embeddings

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        return " ".join(self.vocab.decode_idx_list(batch_input.question)), \
               self.answers[batch_input.answer], \
               self.answers[y_pred]