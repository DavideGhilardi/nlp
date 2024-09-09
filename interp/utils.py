import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

@torch.no_grad()
def get_activations(prompts, model, components, batch_size=8):
    n_layers = len(model.blocks)
    activations = {comp: {i: [] for i in range(n_layers)} for comp in components}
    c = 1 if len(prompts) % batch_size != 0 else 0
    for i in tqdm(range(len(prompts) // batch_size + c)):
        toks = model.to_tokens(prompts[i * batch_size:(i + 1) * batch_size])
        _, cache = model.run_with_cache(toks)

        for comp in components:
            for j in range(n_layers):
                activations[comp][j].append(cache.cache_dict[f'blocks.{j}.{comp}'][:, -1, :].cpu())

        del cache
    for comp in components:
        for j in range(n_layers):
            activations[comp][j] = torch.cat(activations[comp][j], dim=0)

    return activations

def list_to_str(x):
    s = ''
    for i in list(x):
        s += str(i) + ' '

    return s[:-1]

def load_model(model_name, adapter_model="", tl_model="", device='cpu', n_devices=1, dtype=torch.float32):
    print("Loading the model...")
    model = None

    if adapter_model != "":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        peft_model = PeftModel.from_pretrained(model, adapter_model).merge_and_unload()
        del model

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = HookedTransformer.from_pretrained_no_processing(model_name,  hf_model=peft_model, tokenizer=tokenizer,
                                                  device=device, n_devices=n_devices, dtype=dtype)
    else:
        try:
            model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, n_devices=n_devices, dtype=dtype)
            print("Loaded model into HookedTransformer")
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(tl_model)
            model = HookedTransformer.from_pretrained_no_processing(tl_model,  hf_model=model, tokenizer=tokenizer,
                                                    device=device, n_devices=n_devices, dtype=dtype)

    if 'llama' in model_name.lower():
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    model.tokenizer.padding_side = 'left' 

    return model

class FastPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, center=True):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.center = center

    def fit(self, X, y=None):
        # Centering the data
        if self.center:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = torch.zeros(X.shape[-1], device=X.device)
        
        X_centered = X - self.mean_
        
        # Performing SVD
        U, S, Vt = torch.linalg.svd(X_centered.type(torch.float32).to(X.device), full_matrices=False)
        
        # Storing singular values and components
        self.singular_values_ = S[:self.n_components]
        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        # Check if fit has been called
        if self.components_ is None:
            raise RuntimeError("You must fit the model before transforming the data")

        # Centering the data
        X_centered = (X - self.mean_.to(X.device)).type(torch.float32)
        
        # Projecting data onto principal components
        X_pca = X_centered @ self.components_.T.to(X.device)

        return X_pca.cpu()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        pass


def get_train_test(activations, prompts, layer):
    X = activations[layer].numpy()
    y = prompts.apply(lambda row: (row['prompt'], row['labels']), axis=1).to_numpy()
    classes = prompts.labels.to_numpy()
    X_train, X_test, res_y_train, res_y_test = train_test_split(X, y, stratify=classes, random_state=42)
    y_train = np.array([item[1] for item in res_y_train])
    y_test = np.array([item[1] for item in res_y_test])
    prompts_test = np.array([item[0] for item in res_y_test])

    return X_train, X_test, y_train, y_test, prompts_test