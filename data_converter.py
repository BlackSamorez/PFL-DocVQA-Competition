import os
os.environ["OMP_NUM_THREADS"] = "8"

import random
from collections import OrderedDict
from communication.log_communication import log_communication
from communication.compute_tensor_size import get_bytes_for_tensor
from tqdm import tqdm, trange

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.PFL_DocVQA import collate_fn
from transformers import T5Tokenizer

from tqdm import tqdm
from build_utils import (build_dataset, build_model, build_optimizer, build_provider_dataset)
from differential_privacy.dp_utils import (add_dp_noise, clip_parameters, flatten_params, get_shape, reconstruct_shape)
from eval import evaluate  # fl_centralized_evaluation
from logger import Logger
from metrics import Evaluator
from checkpoint import save_model
from utils import load_config, parse_args, seed_everything
from utils_parallel import get_parameters_from_model, set_parameters_model, weighted_average
from collections import OrderedDict

torch.set_num_threads(8)


@torch.no_grad()
def prepare_inputs_for_vqa(tokenizer, visual_embedder, max_input_tokens, question, words, boxes, images, answers=None):
        bs = len(words)
        # input_text = ["question: {:s}  context: {:s}".format(q, c) for q, c in zip(question, words)]
        prompt_text = ["question: {:s}  context: ".format(q) for q in question]
        prompt_box = [0, 0, 1000, 1000]
        eos_box = [0, 0, 0, 0]
        padding_box_value = 0  # To become [0, 0, 0, 0] array.

        # Get input_ids, attention_mask and boxes.
        longest_seq = 0
        batch_input_ids = []
        batch_input_boxes = []
        for batch_idx in range(bs):
            tokenized_prompt = tokenizer(prompt_text[batch_idx])
            input_ids = tokenized_prompt.input_ids[:-1]
            input_boxes = [prompt_box] * len(input_ids)

            for word, box in zip(words[batch_idx], boxes[batch_idx]):
                tokenized_word = tokenizer(word).input_ids[:-1]  # Tokenize the word and ignore eos_token
                input_ids.extend(tokenized_word)
                input_boxes.extend([box]*len(tokenized_word))  # Repeat the box for each token corresponding to the word.

            batch_input_ids.append(input_ids[:max_input_tokens-1] + [tokenizer.eos_token_id])  # Append the eos_token at the end.
            batch_input_boxes.append(np.concatenate([input_boxes[:max_input_tokens-1],  np.array([eos_box])]))  # Append a bounding box corresponding to the eos_token.
            longest_seq = min(max(longest_seq, len(input_ids) + 1), max_input_tokens)

        # Convert to tensors and pad. Actually, a pad tensor is created and it's filled with corresponding values.
        tensor_input_ids = torch.full([bs, longest_seq], fill_value=tokenizer.pad_token_id, dtype=torch.long)
        tensor_boxes = torch.full([bs, longest_seq, 4],  fill_value=padding_box_value, dtype=torch.long)
        tensor_attention_mask = torch.zeros([bs, longest_seq], dtype=torch.long)

        for batch_idx in range(bs):
            tensor_input_ids[batch_idx, :len(batch_input_ids[batch_idx])] = torch.LongTensor(batch_input_ids[batch_idx])
            tensor_boxes[batch_idx, :len(batch_input_boxes[batch_idx])] = torch.from_numpy(batch_input_boxes[batch_idx][:len(batch_input_boxes[batch_idx])])
            tensor_attention_mask[batch_idx, :len(batch_input_ids[batch_idx])] = 1

        # Get semantic and spatial embeddings
        visual_embedding, visual_emb_mask = visual_embedder(images)

        # Tokenize answers
        if answers is not None:
            answers = [random.choice(answer) for answer in answers]
            labels = tokenizer(answers, return_tensors='pt', padding=True)
            labels.input_ids[labels.input_ids[:] == tokenizer.pad_token_id] = -100
            labels = labels.input_ids[0]
        else:
            labels = None

        return tensor_input_ids[0].cpu().numpy(), tensor_boxes[0].cpu().numpy(), tensor_attention_mask[0].cpu().numpy(), visual_embedding[0].cpu().numpy(), visual_emb_mask[0].cpu().numpy(), labels.cpu().numpy()
    
    
if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    seed_everything(config.seed)

    # Set `MASTER_ADDR` and `MASTER_PORT` environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9957'

    visual_embedder = build_model(config).model.visual_embedding
    tokenizer = T5Tokenizer.from_pretrained(config.model_weights)
    
    for i in tqdm(range(10)):
        train_dataset = build_dataset(config, 'train', client_id=i)
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        
        data = [{
            'creation_time': '2023-11-25 14:25:38',
            'version': '2.0.0',
            'split': 'blue_train_client_0',
            'has_answer': True,
        }]
        
        for batch in tqdm(train_data_loader, leave=False):
            question = batch['questions']
            words = batch['words']
            boxes = batch['boxes']
            images = batch['images']
            answers = batch['answers']
            tensor_input_ids, tensor_boxes, tensor_attention_mask, visual_embedding, visual_emb_mask, labels = prepare_inputs_for_vqa(tokenizer, visual_embedder, config.max_input_tokens, question, words, boxes, images, answers)
            
            data.append({
                "tensor_input_ids": tensor_input_ids,
                "tensor_boxes": tensor_boxes,
                "tensor_attention_mask": tensor_attention_mask,
                "visual_embedding": visual_embedding,
                "visual_emb_mask": visual_emb_mask,
                "labels": labels,
                "answers": answers,
            })
            
        np.save(f"/home/black_samorez/diplom/data/nips/preprocessed/train_client_{i}.npy", data, allow_pickle=True)
