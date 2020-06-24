import json
import os
from collections import namedtuple
from typing import Tuple, List, Any

import h5py
import torch
import numpy as np
from dlex import logger
from dlex.datasets.nlp.utils import Vocab, load_embeddings
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.variable_length_tensor import pad_sequence
from ...datatypes import BatchX
from tqdm import tqdm


Sample = namedtuple("Sample", "question_id image_id question answer")


class PytorchGQA(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

        self._vocab = None
        self._answers = None
        self._image_ids = None
        self._image_data = None

        with open(os.path.join(self.processed_data_dir, "spatial_merged_info.json"), "r") as f:
            logger.info("Loading image ids...")
            self.image_ids = {key: val['index'] for key, val in json.load(f).items()}

        if self.use_spatial_features:
            self.h5_spatial = h5py.File(os.path.join(self.processed_data_dir, "spatial.h5"), "r")
            self.image_spatial_features = self.h5_spatial['features']

        if self.use_object_features:
            self.h5_object = h5py.File(os.path.join(self.processed_data_dir, "objects.h5"), "r")
            self.image_object_features = self.h5_object['features']
            self.image_object_bboxes = self.h5_object['bboxes']
            with open(os.path.join(self.processed_data_dir, "objects_merged_info.json"), "r") as f:
                logger.info("Loading object info...")
                self.object_info = json.load(f)

            self.relation_vocab = Vocab.from_file(self.builder.relation_name_path)
            self.object_vocab = Vocab.from_file(self.builder.object_name_path)
            self.attribute_vocab = Vocab.from_file(self.builder.attribute_name_path)
            self.relation_vocab.init_pretrained_embeddings('glove')
            self.object_vocab.init_pretrained_embeddings('glove')
            self.attribute_vocab.init_pretrained_embeddings('glove')

        if self.use_bert_features:
            self.h5_bert = h5py.File(os.path.join(self.processed_data_dir, f"bert_features_{mode}.h5"), "r")
            self.question_bert_outputs = self.h5_bert['outputs']
            self.question_bert_states = self.h5_bert['state']
            self.question_bert_lengths = self.h5_bert['lengths']

    @property
    def use_bert_features(self):
        return self.configs.bert_features

    @property
    def use_object_features(self):
        return self.configs.object_features

    @property
    def use_spatial_features(self):
        return self.configs.spatial_features

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

    def encode_token_list(self, ls: List[str]):
        """
        Encode a sentence
        :param ls: List of word or word id
        :return:
        """
        if all(s.isnumeric() for s in ls):
            return [int(s) for s in ls]
        else:
            return self.vocab.encode_token_list(ls)

    @property
    def data(self):
        if self._data is None:
            self._data = []
            logger.info("Loading from %s...", self.builder.get_data_path(self.mode))
            with open(self.builder.get_data_path(self.mode)) as f:
                lines = [l for l in f.read().split('\n') if l.strip() != ""]
            answer_dict = {a: i for i, a in enumerate(self.answers)}
            for line in tqdm(lines[:self.configs.size or -1], leave=False, desc="Parse dataset"):
                qid, image_id, q, a = line.split('\t')
                self._data.append(Sample(
                    question_id=qid,
                    image_id=image_id,
                    question=self.encode_token_list(q.split(' ')),
                    answer=answer_dict.get(a, -1)))

            # assert all(len(self.answers) > s.answer >= 0 for s in self._data)
            logger.info(f"Dataset loaded. Number of samples: {len(lines):,}")
            logger.info(
                "Question length - max: %d - avg: %d",
                max(len(s.question) for s in self._data),
                np.average([len(s.question) for s in self._data]))
        return self._data

    @property
    def image_data(self):
        if self._image_data is None:
            self._image_data = {}
            with open(self.builder.get_image_data_path(self.mode)) as f:
                lines = [l for l in f.read().strip().split('\n')]
            for line in tqdm(lines):
                img_id, obj_id, identity, attributes, relations = line.split('\t')
                if img_id not in self._image_data:
                    self._image_data[img_id] = [[], [], []]
                self.image_data[img_id][0].append(self.object_vocab.encode_token_list(identity.split(' ')))
                self.image_data[img_id][1].append([self.attribute_vocab.encode_token_list(attr.split(' ')) for attr in attributes.split(',')])
                self.image_data[img_id][2].append([self.relation_vocab.encode_token_list(rela.split(' ')) for rela in relations.split(',')])

        return self._image_data

    def get_attribute_name_embeddings(self):
        return torch.FloatTensor(self.attribute_vocab.embeddings)

    def get_concept_embeddings(self):
        return torch.FloatTensor(self.attribute_vocab.embeddings)

    def close(self):
        self.h5_spatial.close()
        self.h5_object.close()

    def __getitem__(self, i):
        if type(i) == int:
            _i = slice(i, i + 1)
        else:
            _i = i
        samples = self.data[_i]

        if self.use_spatial_features:
            img = [torch.from_numpy(self.image_spatial_features[self.image_ids[s.image_id]]) for s in samples]
        else:
            img = [None for _ in range(len(samples))]

        if self.use_object_features:
            obj = []
            obj_bboxes = []
            obj_identities = []
            obj_relations = []
            obj_attributes = []
            for s in samples:
                img_id = self.image_ids[s.image_id]
                obj.append(self.image_object_features[img_id][:self.object_info[s.image_id]['objectsNum']].tolist())
                obj_bboxes.append(self.image_object_bboxes[img_id])

                obj_identities, obj_attributes, obj_relations = self._image_data[s.image_id]
                print(obj_identities)
                input()

        else:
            obj = obj_bboxes = [None for _ in range(len(samples))]

        q = [s.question for s in samples]
        qid = [s.question_id for s in samples]
        ans = [s.answer for s in samples]

        if self.use_bert_features:
            bert_outputs = self.question_bert_outputs[_i]
            bert_states = self.question_bert_states[_i]
            bert_lengths = self.question_bert_lengths[_i]
        else:
            bert_outputs = bert_states = bert_lengths = [None for _ in range(len(samples))]

        ret = list(zip(
            img,
            obj,
            obj_bboxes,
            obj_identities,
            obj_attributes,
            obj_relations,
            q,
            qid,
            bert_outputs,
            bert_states,
            bert_lengths,
            ans))
        return ret[0] if type(i) == int else ret

    def collate_fn(self, batch: List[Tuple]):
        batch = sorted(batch, key=lambda x: len(x[3]), reverse=True)
        imgs, objs, obj_bboxes, obj_identities, obj_attributes, obj_relations, \
            qs, qids, bert_outputs, bert_states, bert_lengths, ans \
            = [[b[i] for b in batch] for i in range(len(batch[0]))]

        qs, qlen = pad_sequence(qs, self.vocab.blank_token_idx, output_tensor=True)

        if self.use_bert_features:
            bert_max_length = max(bert_lengths)

        if self.use_object_features:
            objs, objlen = pad_sequence(objs, 0., output_tensor=True)

        batch_x = BatchX(
            questions=qs,
            question_lengths=qlen,
            images=torch.stack(imgs) if self.use_spatial_features else None,
            objects=objs if self.use_object_features else None,
            object_bounding_boxes=torch.FloatTensor(obj_bboxes) if self.use_object_features else None,
            object_lengths=objlen if self.use_object_features else None,
            question_bert_states=torch.FloatTensor(bert_states) if self.use_bert_features else None,
            question_bert_outputs=torch.FloatTensor([o[:bert_max_length] for o in bert_outputs]) if self.use_bert_features else None,
            question_bert_lengths=torch.LongTensor(bert_lengths) if self.use_bert_features else None,
            object_identities=obj_identities if self.use_object_features else None,
            object_attributes=obj_attributes if self.use_object_features else None,
            object_relations=obj_relations if self.use_object_features else None)

        return Batch(
            ids=qids,
            X=batch_x,
            Y=torch.LongTensor(ans))

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        # " ".join(self.vocab.decode_idx_list(batch_input.X[2][:batch_input.X[3]])),
        return "", \
               self.answers[y_pred], \
               self.answers[batch_input.Y]

    def shuffle(self):
        raise Exception("This dataset cannot be shuffled")

    def write_results_to_file(
            self,
            all_predictions: List[Any],
            sample_ids: List[Any],
            output_path: str,
            output_tag: str,
            format: str = None) -> str:
        os.makedirs(output_path, exist_ok=True)
        res = [dict(
            questionId=qid,
            prediction=self.answers[pred]
        ) for qid, pred in zip(sample_ids, all_predictions)]
        path = os.path.join(output_path, output_tag + ".json")
        with open(path, "w") as f:
            json.dump(res, f)
            logger.info("Results written to %s", path)
        return path
