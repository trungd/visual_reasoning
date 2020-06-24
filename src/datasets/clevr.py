import json
import os

import h5py
import numpy as np
import torch
import torchvision
from PIL import Image
from dlex import Params
from dlex.datasets import DatasetBuilder
from dlex.datasets.builder import ModelStringOutput
from dlex.datasets.nlp.utils import write_vocab, nltk_tokenize
from dlex.utils import logger
from imageio import imread
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from torchvision.models import resnet101, ResNet
from torchvision.transforms import Resize
from tqdm import tqdm


def tokenize(s: str) -> str:
    s = s.lower()
    s = ''.join([c for c in s if c not in list("?;")])
    return nltk_tokenize(s)


class _CLEVRImage(TorchDataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.length = len(os.listdir(os.path.join(root, 'images', mode)))
        self.transform = transforms.Compose([
            Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img = os.path.join(
            self.root, 'images', self.mode,
            'CLEVR_{}_{}.png'.format(self.mode, str(index).zfill(6)))
        img = Image.open(img).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return self.length


class CLEVR(DatasetBuilder):
    def __init__(self, params: Params):
        super().__init__(params, [
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0_no_images.zip",
        ])
        self._resnet = None
        self._tf_resnet = None
        self._answer_dict = []

    @property
    def answer_path(self):
        return os.path.join(self.get_processed_data_dir(), "answers.txt")

    @property
    def vocab_path(self):
        return os.path.join(self.get_processed_data_dir(), "vocab.txt")

    @property
    def torch_resnet(self):
        if not self._resnet:
            logger.info("Initializing Resnet...")
            resnet = resnet101(True).cuda()
            resnet.eval()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                return x

            resnet.forward = forward.__get__(resnet, ResNet)
            self._resnet = resnet
        return self._resnet

    @property
    def tf_resnet(self):
        if not self._tf_resnet:
            cfg = self.params.model.image
            cnn = getattr(torchvision.models, cfg.model)(pretrained=True)
            layers = [
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
            ]
            for i in range(cfg.model_stage):
                name = 'layer%d' % (i + 1)
                layers.append(getattr(cnn, name))
            model = torch.nn.Sequential(*layers)
            model.cuda()
            model.eval()
            self._tf_resnet = model
        return self._tf_resnet

    def maybe_preprocess(self, force=False):
        if not super().maybe_preprocess(force):
            return

        for mode in ["train", "test", "val"]:
            logger.info(f"Loading JSON ({mode})...")
            with open(os.path.join(
                    self.get_raw_data_dir(), 'CLEVR_v1.0', "questions",
                    f'CLEVR_{mode}_questions.json')) as f:
                data = json.load(f)

            self.process_questions(mode, data)
            # self.tf_process_image_features(mode)
            # self.torch_process_image_features(mode, len(data['questions']))

    def torch_process_image_features(self, mode, size):
        logger.info("Extracting image features...")
        batch_size = 50
        dataset = _CLEVRImage(os.path.join(self.get_raw_data_dir(), 'CLEVR_v1.0'), mode=mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        with h5py.File(self.get_image_features_path(mode, "torch"), "w", libver='latest') as f:
            dset = f.create_dataset('data', (size, 1024, 14, 14), dtype='f4')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                for i, img in tqdm(
                        enumerate(dataloader),
                        total=len(dataset) // batch_size,
                        desc="Extract image features"):
                    img = img.to(device)
                    features = self.torch_resnet(img).detach().cpu().numpy()
                    start = i * batch_size
                    end = min(size, (i + 1) * batch_size)
                    dset[start:end] = features

    def tf_process_image_features(self, mode):
        def run_batch(cur_batch, model):
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

            image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
            image_batch = (image_batch / 255.0 - mean) / std
            image_batch = torch.FloatTensor(image_batch).cuda()
            with torch.no_grad():
                image_batch = torch.autograd.Variable(image_batch)

            feats = model(image_batch)
            feats = feats.data.cpu().clone().numpy()

            return feats

        logger.info("Extracting image features...")
        cfg = self.params.model.image
        input_paths = []
        idx_set = set()
        input_image_dir = os.path.join(self.get_raw_data_dir(), 'CLEVR_v1.0', "images", mode)
        for fn in os.listdir(input_image_dir):
            if not fn.endswith('.png'):
                continue
            idx = int(os.path.splitext(fn)[0].split('_')[-1])
            input_paths.append((os.path.join(input_image_dir, fn), idx))
            idx_set.add(idx)
        input_paths.sort(key=lambda x: x[1])
        assert len(idx_set) == len(input_paths)
        assert min(idx_set) == 0 and max(idx_set) == len(idx_set) - 1
        logger.info(f"No. samples: {len(idx_set)}")

        img_size = (cfg.height, cfg.width)
        with h5py.File(self.get_image_features_path(mode, "tf"), 'w') as f:
            feat_dset = None
            i0 = 0
            cur_batch = []
            for i, (path, idx) in tqdm(list(enumerate(input_paths)), desc="Process images", leave=False):
                img = imread(path, pilmode='RGB')
                img = Image.fromarray(img).resize(img_size, Image.BICUBIC)
                img = np.array(img)
                img = img.transpose([2, 0, 1])[None]
                cur_batch.append(img)
                if len(cur_batch) == self.params.train.batch_size:
                    feats = run_batch(cur_batch, self.tf_resnet)
                    if feat_dset is None:
                        N = len(input_paths)
                        _, C, H, W = feats.shape
                        feat_dset = f.create_dataset('features', (N, C, H, W), dtype=np.float32)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats
                    i0 = i1
                    cur_batch = []
                elif i == len(input_paths) - 1:
                    feats = run_batch(cur_batch, self.tf_resnet)
                    i1 = i0 + len(cur_batch)
                    feat_dset[i0:i1] = feats

    def process_questions(self, mode, data):
        questions = []
        answers = []
        image_filenames = []

        for question in tqdm(data['questions'], desc=f"Tokenizing {mode}"):
            questions.append(tokenize(question['question']))
            answers.append(question.get('answer', "none").strip())
            image_filenames.append(question['image_filename'])

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

    def get_pytorch_wrapper(self, mode: str):
        from ..torch.datasets.clevr import PytorchCLEVR
        return PytorchCLEVR(self, mode)

    def get_tensorflow_wrapper(self, mode: str):
        from ..tf.datasets.clevr import TensorflowCLEVR
        return TensorflowCLEVR(self, mode)

    def get_image_features_path(self, mode: str, tag: str):
        if mode == "valid":
            mode = "val"
        return os.path.join(self.get_processed_data_dir(), f"{mode}_{tag}_features.hdf5")

    def get_data_path(self, mode: str):
        if mode == "valid":
            mode = "val"
        return os.path.join(self.get_processed_data_dir(), f"{mode}.csv")

    def format_output(self, y_pred, batch_item) -> ModelStringOutput:
        return ModelStringOutput(None, None, y_pred)

