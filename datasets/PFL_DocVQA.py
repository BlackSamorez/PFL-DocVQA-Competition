import os, random
from PIL import Image
from torch.utils.data import Dataset
import torch

import numpy as np


class PFL_DocVQA(Dataset):

    def __init__(self, imbd_dir, images_dir, split, kwargs, indexes=None):

        if 'client_id' not in kwargs:
            data = np.load(os.path.join(imbd_dir, "imdb_{:s}.npy".format(split)), allow_pickle=True)
        else:
            data = np.load(os.path.join(imbd_dir, "imdb_{:s}_client_{:}.npy".format(split, kwargs['client_id'])), allow_pickle=True)

        # keep only data points of given provider
        if indexes:
            selected = [0] + indexes
            data = [data[i] for i in selected]

        self.header = data[0]
        self.imdb = data[1:]
        self.images_dir = images_dir

        self.use_images = kwargs.get('use_images', False)
        self.get_raw_ocr_data = kwargs.get('get_raw_ocr_data', False)

    def __len__(self):
        return len(self.imdb)  #  min(20, len(self.imdb))

    def __getitem__(self, idx):
        record = self.imdb[idx]

        question = record["question"]
        answers = [record['answers'].lower()]
        context = " ".join([word.lower() for word in record['ocr_tokens']])

        if self.get_raw_ocr_data:
            if len(record['ocr_tokens']) == 0:
                words = []
                boxes = np.empty([0, 4])
                                 
            else:
                words = [word.lower() for word in record['ocr_tokens']]
                boxes = np.array([bbox for bbox in record['ocr_normalized_boxes']])

        if self.use_images:
            image_names = os.path.join(self.images_dir, "{:s}.jpg".format(record['image_name']))
            images = Image.open(image_names).convert("RGB")

        sample_info = {
            'question_id': record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx)),
            'questions': question,
            'contexts': context,
            'answers': answers,
        }

        if self.use_images:
            sample_info['image_names'] = image_names
            sample_info['images'] = images

        if self.get_raw_ocr_data:
            sample_info['words'] = words
            sample_info['boxes'] = boxes

        return sample_info


def collate_fn(batch):
    batch = {k: [dic[k] for dic in batch] for k in batch[0]}  # List of dictionaries to dict of lists.
    return batch


class PFL_DocVQA_preprocessed(Dataset):
    pad_token_id: int = 0
    padding_box_value: int = 0
    def __init__(self, data_dir, split, kwargs, indexes=None):
        if 'client_id' not in kwargs:
            data = np.load(os.path.join(data_dir, "{:s}.npy".format(split)), allow_pickle=True)
        else:
            data = np.load(os.path.join(data_dir, "{:s}_client_{:}.npy".format(split, kwargs['client_id'])), allow_pickle=True)

        # keep only data points of given provider
        if indexes:
            selected = [0] + indexes
            data = [data[i] for i in selected]

        self.header = data[0]
        rows = data[1:]
        self.rows = []
        for row in rows:
            new_row = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    new_row[k] = torch.from_numpy(v)
                else:
                    new_row[k] = v
                self.rows.append(new_row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn_preprocessed(batch):
    collated_batch = {}
    collated_batch["tensor_input_ids"] = torch.nn.utils.rnn.pad_sequence([element["tensor_input_ids"] for element in batch], batch_first=True, padding_value=PFL_DocVQA_preprocessed.pad_token_id)
    collated_batch["tensor_boxes"] = torch.nn.utils.rnn.pad_sequence([element["tensor_boxes"] for element in batch], batch_first=True, padding_value=PFL_DocVQA_preprocessed.padding_box_value)
    collated_batch["tensor_attention_mask"] = torch.nn.utils.rnn.pad_sequence([element["tensor_attention_mask"] for element in batch], batch_first=True, padding_value=0)
    collated_batch["visual_embedding"] = torch.nn.utils.rnn.pad_sequence([element["visual_embedding"] for element in batch], batch_first=True, padding_value=0)
    collated_batch["visual_emb_mask"] = torch.nn.utils.rnn.pad_sequence([element["visual_emb_mask"] for element in batch], batch_first=True, padding_value=0)
    collated_batch["labels"] = torch.nn.utils.rnn.pad_sequence([element["labels"] for element in batch], batch_first=True, padding_value=0) if batch[0]["labels"] is not None else None
    collated_batch["answers"] = ["yes"] * len(collated_batch["tensor_input_ids"])
    return collated_batch
