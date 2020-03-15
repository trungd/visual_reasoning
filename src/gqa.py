import json
import os
import re
import subprocess

import h5py
from dlex import Params, logger, List
from dlex.datasets import DatasetBuilder
from dlex.datasets.nlp.utils import write_vocab, nltk_tokenize
from tqdm import tqdm

from .tf.gqa import TensorflowGQA


def tokenize(s: str) -> List[str]:
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
                "https://nlp.stanford.edu/data/gqa/allImages.zip",
                "https://nlp.stanford.edu/data/gqa/eval.zip"
            ], tensorflow_cls=TensorflowGQA)

    @property
    def answer_path(self):
        fn = "answers"
        if self.configs.balanced:
            fn += "_balanced"
        return os.path.join(self.get_processed_data_dir(), f"{fn}.txt")

    @property
    def vocab_path(self):
        fn = "vocab"
        if self.configs.balanced:
            fn += "_balanced"
        return os.path.join(self.get_processed_data_dir(), f"{fn}.txt")

    def get_data_path(self, mode: str):
        if mode == "valid":
            mode = "val"
        fn = mode
        if self.configs.balanced:
            fn += "_balanced"
        return os.path.join(self.get_processed_data_dir(), f"{fn}.csv")

    def get_image_features_path(self):
        return os.path.join(self.get_processed_data_dir(), "spatial.h5")

    def maybe_preprocess(self, force=False):
        if not super().maybe_preprocess(force):
            return

        for mode in ["train", "val", "testdev", "test", "submission", "challenge"]:
            data = self.load_json_data(mode)
            self.process_questions(mode, data)

        # self.tf_process_image_features(mode)
        # self.torch_process_image_features(mode, len(data['questions']))
        # self.merge_features("spatial")
        # self.merge_features("objects")

    def load_json_data(self, mode):
        logger.info(f"Loading {mode} data...")
        if mode == "train":
            root_path = os.path.join(self.get_raw_data_dir(), f'{mode}_all_questions')
            paths = [os.path.join(root_path, fn) for fn in sorted(os.listdir(root_path))]
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
                        "bboxes": (148855, 100, 4)}}

        lengths = [0]
        with h5py.File(os.path.join(self.get_processed_data_dir(), f"{name}.h5"), "w") as out:
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
            with open(os.path.join(self.get_processed_data_dir(), f"{name}_merged_info.json"), "w") as infoOut:
                json.dump(info, infoOut)

    def process_questions(self, mode, data):
        questions = []
        answers = []
        image_filenames = []

        for key in tqdm(data, desc=f"Tokenizing {mode}"):
            if self.configs.balanced:
                if not data[key]['isBalanced']:
                    continue
            questions.append(tokenize(data[key]['question']))
            answers.append(data[key].get('answer', "none").strip())
            image_filenames.append(data[key]['imageId'])

        if mode == "train":
            answer_list = list(set(answers))
            with open(self.answer_path, "w") as f:
                f.write("\n".join(answer_list))
            write_vocab(questions, self.vocab_path)

        with open(self.get_data_path(mode), 'w') as f:
            for qid, q, a, img in zip(
                    list(data.keys()),
                    questions,
                    answers,
                    image_filenames):
                f.write('\t'.join([
                    qid,
                    img,
                    ' '.join(q),
                    a,
                ]) + "\n")

    def run_evaluation_script(
            self,
            result_path,
            **kwargs):
        cmd = [
            "python", os.path.join(self.get_raw_data_dir(), "eval.py"),
            "--tier", "val",
            "--scenes", os.path.join(self.get_raw_data_dir(), "val_sceneGraphs.json"),
            "--choices", os.path.join(self.get_raw_data_dir(), "val_choices.json"),
            "--predictions", result_path,
            "--questions", os.path.join(self.get_raw_data_dir(), "val_all_questions.json")
        ]
        logger.info("Running evaluation script...\n%s", " ".join(cmd))
        res = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        res, _ = res.communicate()
        res = res.decode()
        logger.info("Evaluation script output:\n%s", res)
        return dict(
            accuracy=float(re.search(r"Accuracy: (\d*\.\d*)%", res).group(1)),
            validity=float(re.search(r"Validity: (\d*\.\d*)%", res).group(1)),
            plausibility=float(re.search(r"Plausibility: (\d*\.\d*)%", res).group(1)),
            distribution=float(re.search(r"Distribution: (\d*\.\d*)", res).group(1))
        )