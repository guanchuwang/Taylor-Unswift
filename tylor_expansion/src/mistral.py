from typing import Dict, List, Optional, Set, Tuple, Union
import torch
from torch import nn
import ipdb, re
# from transformers.models.phi.modeling_phi import MistralMLP, MistralDecoderLayer
from .utils import silu_grad
from .utils import model_debug as base_model_debug
from .utils import model_tylor_expansion as base_model_tylor_expansion
from .utils import hiddenstates_benchmark as base_hiddenstates_benchmark
from .llama import LlamaMLP_, Tylor_expansion_LlamaMLP


class MistralMLP_(LlamaMLP_):

    def __init__(self, llama_mlp, config, layer_index=0) -> None:
        super().__init__(llama_mlp, config, layer_index=layer_index)


class Tylor_expansion_MistralMLP(Tylor_expansion_LlamaMLP):

    def __init__(self, mistral_mlp, config, grad_order=6, select_dim=4, max_gap=0,
                 hidden_states_mean=None, hidden_states_std=None, hidden_states_max=None,
                 hidden_states_min=None, grad_order_min=4, delta_hidden_state_thd=2.5,
                 bitwidth=1, layer_index=0) -> None:
        super().__init__(mistral_mlp, config, grad_order, select_dim, max_gap,
                         hidden_states_mean, hidden_states_std, hidden_states_max,
                         hidden_states_min, grad_order_min, delta_hidden_state_thd, bitwidth, layer_index)

    def act_grad(self, x, grad_order, dtype=torch.bfloat16):
        return silu_grad(x, grad_order, dtype=dtype)


def hiddenstates_benchmark(model, fname):
    return base_hiddenstates_benchmark(model, fname)


def model_debug(model):
    return base_model_debug(model, MistralMLP_)


def model_tylor_expansion(model,
                          param_fname=None,
                          expand_layer=None,
                          select_dim=256,
                          grad_order=12,
                          max_gap=0,
                          grad_order_min=4,
                          delta_hidden_state_thd=2.5):
    # print("xxxx")
    return base_model_tylor_expansion(model,
                                      Tylor_expansion_MistralMLP,
                                      param_fname,
                                      expand_layer,
                                      select_dim=select_dim,
                                      grad_order=grad_order,
                                      max_gap=max_gap,
                                      grad_order_min=grad_order_min,
                                      delta_hidden_state_thd=delta_hidden_state_thd)


def model_power_forward(model):
    for layer_index in range(len(model.model.layers)):
        model.model.layers[layer_index].mlp.forward = model.model.layers[layer_index].mlp.forward_power






