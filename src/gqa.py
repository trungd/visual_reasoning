import json
import os

import h5py
from dlex import Params, logger
from dlex.datasets import DatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, nltk_tokenize
from .tf.gqa import TensorflowGQA
from tqdm import tqdm


def tokenize(s: str) -> str:
    s = s.lower()
    s = ''.join([c for c in s if c not in list("?;")])
    return nltk_tokenize(s)


class GQA(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(
            params,
            [
                "https://nlp.stanford.edu/data/gqa/sceneGraphs.zip",
                "https://nlp.stanford.edu/data/gqa/questions1.2.zip",
                "https://nlp.stanford.edu/data/gqa/allImages.zip"
            ], tensorflow_cls=TensorflowGQA)

    @property
    def answer_path(self):
        return os.path.join(self.get_processed_data_dir(), "answers.txt")

    @property
    def vocab_path(self):
        return os.path.join(self.get_processed_data_dir(), "vocab.txt")

    def get_data_path(self, mode: str):
        if mode == "valid":
            mode = "val"
        return os.path.join(self.get_processed_data_dir(), f"{mode}.csv")

    def get_image_features_path(self, mode: str):
        return os.path.join(self.get_raw_data_dir(), "gqa_spatial.h5")

    def maybe_preprocess(self, force=False):
        # if not super().maybe_preprocess(force):
        #     return

        # for mode in ["train", "val", "testdev", "test"]:
        #     data = self.load_json_data(mode)
        #     self.process_questions(mode, data)

        # self.tf_process_image_features(mode)
        # self.torch_process_image_features(mode, len(data['questions']))
        self.merge_features("spatial")

    def load_json_data(self, mode):
        logger.info(f"Loading {mode} data...")
        if mode == "train":
            root_path = os.path.join(self.get_raw_data_dir(), f'{mode}_all_questions')
            paths = [os.path.join(root_path, fn) for fn in os.listdir(root_path)]
        else:
            paths = [os.path.join(self.get_raw_data_dir(), f"{mode}_all_questions.json")]

        logger.info(f"Loading JSON...")
        data = {}
        for path in tqdm(paths, desc="Loading json"):
            with open(path) as f:
                data = {**data, **json.load(f)}
        return data

    def merge_features(self, name, num_chunks=16):
        logger.info(f"Merging dataset...")
        spec = {
            "spatial": {"features": (148855, 2048, 7, 7)},
            "objects": {"features": (148855, 100, 2048),
                        "bboxes": (148855, 100, 4)}
        }

        lengths = [0]
        with h5py.File(os.path.join(self.get_processed_data_dir(), f"gqa_{name}.h5"), "w") as out:
            datasets = {}
            for dname in spec[name]:
                datasets[dname] = out.create_dataset(dname, spec[name][dname])

            low = 0
            for i in tqdm(range(num_chunks), desc="Merge dataset"):
                with h5py.File(os.path.join(self.get_raw_data_dir(), name, f"gqa_{name}_{i}.h5"), "r") as chunk:
                    high = low + chunk["features"].shape[0]
                    for dname in spec[name]:
                        datasets[dname][low:high] = chunk[dname][:]
                    low = high
                    lengths.append(high)

        logger.info(f"Saving {name} info...")
        with open(os.path.join(self.get_raw_data_dir(), name, f"gqa_{name}_info.json")) as infoIn:
            info = json.load(infoIn)
            for imageId in info:
                info[imageId]["index"] = lengths[info[imageId]["file"]] + info[imageId]["idx"]
                del info[imageId]["idx"]
                del info[imageId]["file"]
            with open(os.path.join(self.get_raw_data_dir(), f"gqa_{name}_merged_info.json", "w")) as infoOut:
                json.dump(info, infoOut)

    def process_questions(self, mode, data):
        questions = []
        answers = []
        image_filenames = []

        for key in tqdm(data, desc=f"Tokenizing {mode}"):
            questions.append(tokenize(data[key]['question']))
            answers.append(data[key].get('answer', "none").strip())
            image_filenames.append(data[key]['imageId'])

        if mode == "train":
            self._answer_dict = {ans: i for i, ans in enumerate(set(answers))}
            with open(self.answer_path, "w") as f:
                f.write("\n".join(list(set(answers))))

        with open(self.get_data_path(mode), 'w') as f:
            for q, a, img in zip(questions, answers, image_filenames):
                f.write('\t'.join([
                    img,
                    ' '.join(q),
                    a,
                    str(self._answer_dict.get(a, -1))
                ]) + "\n")

        if mode == "train":
            write_vocab(questions, self.vocab_path)