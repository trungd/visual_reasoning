import json
import os
from collections import namedtuple
from typing import Dict, Any

import h5py
import tensorflow.compat.v1 as tf
import numpy as np
from dlex import logger, List
from dlex.datasets.nlp.utils import Vocab
from dlex.datasets.tf import Dataset
from dlex.tf.utils.utils import pad_sequence
from tqdm import tqdm

Sample = namedtuple("Sample", "question_id image_id question answer")
Placeholders = namedtuple("Placeholders", "images questions question_lengths answers")


class TensorflowGQA(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

        # Lazy loading
        self._vocab = None
        self._answers = None
        self._image_ids = None

        self.h5_spatial = h5py.File(os.path.join(self.processed_data_dir, "spatial.h5"), 'r')
        self.image_spatial_features = self.h5_spatial['features']
        # self.image_idx = self.h['indices']
        self._word_embeddings = None

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = Vocab.from_file(self.builder.vocab_path)
            logger.info("Vocab size: %d", len(self._vocab))
        return self._vocab

    @property
    def answers(self):
        if self._answers is None:
            self._answers = open(self.builder.answer_path, "r").read().strip().split('\n')
            logger.info("Number of answers: %d", len(self._answers))
        return self._answers

    @property
    def num_classes(self):
        return len(self.answers)

    @property
    def data(self):
        if self._data is None:
            self._data = []
            logger.info("Loading from %s...", self.builder.get_data_path(self.mode))
            with open(self.builder.get_data_path(self.mode)) as f:
                lines = [l for l in f.read().split('\n') if l.strip() != ""]
            answer_dict = {a: i for i, a in enumerate(self.answers)}
            for line in tqdm(lines, leave=False, desc="Parse dataset"):
                qid, image_id, q, a = line.split('\t')
                self._data.append(Sample(
                    question_id=qid,
                    image_id=image_id,
                    question=self.vocab.encode_token_list(q.split(' ')),
                    answer=answer_dict.get(a, -1)))

            assert all(len(self.answers) > s.answer >= 0 for s in self._data)
            logger.info(f"Dataset loaded. Number of samples: {len(lines):,}")
            logger.info(
                "Question length - max: %d - avg: %d",
                max(len(s.question) for s in self._data),
                np.average([len(s.question) for s in self._data]))
        return self._data

    @property
    def image_ids(self):
        if self._image_ids is None:
            with open(os.path.join(self.processed_data_dir, "spatial_merged_info.json"), "r") as f:
                self._image_ids = {key: val['index'] for key, val in json.load(f).items()}
        return self._image_ids

    def close(self):
        self.h5_spatial.close()

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
        feed_dict[placeholders.images] = [self.image_spatial_features[self.image_ids[s.image_id]] for s in data]
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

    def write_results_to_file(
            self,
            all_predictions: List[Any],
            output_path: str,
            output_tag: str,
            format: str = None) -> str:
        os.makedirs(output_path, exist_ok=True)
        res = [dict(
            questionId=sample.question_id,
            prediction=self.answers[pred]
        ) for sample, pred in zip(self.data, all_predictions)]
        path = os.path.join(output_path, output_tag + ".json")
        with open(path, "w") as f:
            json.dump(res, f)
            logger.info("Results written to %s", path)
        return path
