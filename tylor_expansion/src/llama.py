from typing import Dict, List, Optional, Set, Tuple, Union
import torch
from torch import nn
import ipdb, re
import math
# from transformers.models.llama.modeling_llama import LlamaMLP, LlamaDecoderLayer
from .utils import silu_grad, dim_selection, min_dim_selection
from .utils import model_debug as base_model_debug
from .utils import model_tylor_expansion as base_model_tylor_expansion
from .utils import hiddenstates_benchmark as base_hiddenstates_benchmark


# def gaussian_cdf(x):
#     return (1 + torch.erf(x/math.sqrt(2))) * 0.5
#     # return (torch.nn.functional.gelu(x) + 1e-20) / (x + 1e-20)
#
# def gaussian_pdf(x):
#     return 1/math.sqrt(2 * torch.pi) * torch.exp(- x**2 / 2)


class LlamaMLP_(nn.Module):

    def __init__(self, llama_mlp, config, layer_index=0) -> None:
        super().__init__()

        self.config = config
        self.gate_proj = llama_mlp.gate_proj
        self.up_proj = llama_mlp.up_proj
        self.down_proj = llama_mlp.down_proj
        self.act_fn = llama_mlp.act_fn

        self.intermediate_size, self.hidden_size = llama_mlp.gate_proj.weight.data.shape
        self.hidden_states_intermediate = torch.zeros(self.intermediate_size, dtype=self.up_proj.weight.dtype,
                                                      device=self.up_proj.weight.device)  # None
        self.hidden_states_intermediate_mean = torch.zeros(self.intermediate_size, dtype=self.up_proj.weight.dtype,
                                                           device=self.up_proj.weight.device)  # 0
        self.hidden_states_intermediate_square_mean = torch.zeros(self.intermediate_size,
                                                                  dtype=self.up_proj.weight.dtype,
                                                                  device=self.up_proj.weight.device)  # 0
        self.hidden_states_intermediate_std = torch.zeros(self.intermediate_size, dtype=self.up_proj.weight.dtype,
                                                          device=self.up_proj.weight.device)  # 0
        self.hidden_states_intermediate_max = torch.zeros(self.intermediate_size, dtype=self.up_proj.weight.dtype,
                                                          device=self.up_proj.weight.device)  # None
        self.hidden_states_intermediate_min = torch.zeros(self.intermediate_size, dtype=self.up_proj.weight.dtype,
                                                          device=self.up_proj.weight.device)  # None
        self.batch_num = 0
        self.layer_index = layer_index
        print(layer_index)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        self.batch_num = self.batch_num + 1
        hidden_states_intermediate = (hidden_states @ self.gate_proj.weight.T)
        hidden_states_intermediate = hidden_states_intermediate.view(-1, hidden_states_intermediate.shape[-1])

        self.hidden_states_intermediate_mean = (
                                                           self.batch_num - 1) * 1. / self.batch_num * self.hidden_states_intermediate_mean + \
                                               hidden_states_intermediate.sum(0) / self.batch_num / \
                                               hidden_states_intermediate.shape[0]

        hidden_states_intermediate_square = hidden_states_intermediate.square()
        self.hidden_states_intermediate_square_mean = (
                                                                  self.batch_num - 1) * 1. / self.batch_num * self.hidden_states_intermediate_square_mean + \
                                                      hidden_states_intermediate_square.sum(0) / self.batch_num / \
                                                      hidden_states_intermediate_square.shape[0]

        self.hidden_states_intermediate_std = torch.sqrt(
            self.hidden_states_intermediate_square_mean - self.hidden_states_intermediate_mean.square())

        hidden_states_intermediate_max = hidden_states_intermediate.max(0).values
        if self.hidden_states_intermediate_max is None:
            self.hidden_states_intermediate_max = hidden_states_intermediate_max
        else:
            max_index = hidden_states_intermediate_max > self.hidden_states_intermediate_max
            self.hidden_states_intermediate_max[max_index] = hidden_states_intermediate_max[max_index]

        hidden_states_intermediate_min = hidden_states_intermediate.min(0).values
        if self.hidden_states_intermediate_min is None:
            self.hidden_states_intermediate_min = hidden_states_intermediate_min
        else:
            min_index = hidden_states_intermediate_min < self.hidden_states_intermediate_min
            self.hidden_states_intermediate_min[min_index] = hidden_states_intermediate_min[min_index]

        # print(self.batch_num)
        # if self.layer_index == 0:
        #     print(hidden_states_intermediate[:, 1128])
        #     print(self.hidden_states_intermediate_mean[1128])
        #     ipdb.set_trace()

        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        output = self.down_proj(self.act_fn(gate_output) * up_output)

        # ipdb.set_trace()
        # print(self.layer_index)

        return output


class Tylor_expansion_LlamaMLP(nn.Module):
    def __init__(self, llama_mlp, config, grad_order=6, select_dim=4, max_gap=10,
                 hidden_states_mean=None, hidden_states_std=None, hidden_states_max=None,
                 hidden_states_min=None, grad_order_min=4, delta_hidden_state_thd=2.5, bitwidth=1,
                 layer_index=0) -> None:
        super().__init__()

        self.config = config
        self.up_proj = llama_mlp.up_proj
        self.down_proj = llama_mlp.down_proj
        self.gate_proj = llama_mlp.gate_proj
        self.act_fn = llama_mlp.act_fn
        self.layer_index = layer_index

        self.intermediate_size, self.hidden_size = llama_mlp.gate_proj.weight.data.shape
        self.hidden_states_mean = hidden_states_mean.to(self.up_proj.weight.device)
        self.hidden_states_std = hidden_states_std.to(self.up_proj.weight.device)
        self.hidden_states_max = hidden_states_max.to(self.up_proj.weight.device)
        self.hidden_states_min = hidden_states_min.to(self.up_proj.weight.device)
        self.max_min_gap = self.hidden_states_max - self.hidden_states_min

        # self.max_min_gap = torch.randn(self.intermediate_size)

        # std_sort_index = (-self.max_min_gap).argsort()
        # sampled_dim = std_sort_index[:select_dim]
        # ffn_dim = std_sort_index[select_dim:]

        # sampled_dim, ffn_dim = min_dim_selection(self.max_min_gap, select_dim)
        sampled_dim, ffn_dim = dim_selection(self.max_min_gap, select_dim, max_gap=max_gap)

        self.sampled_dim = sampled_dim
        self.ffn_dim = ffn_dim
        print("layer {}, Max min gap {}, {}, {}".format(layer_index, self.max_min_gap[self.sampled_dim].min(),
                                                        self.max_min_gap[self.sampled_dim].median(),
                                                        self.max_min_gap[self.sampled_dim].max()))

        self.local_hidden_states_min = self.hidden_states_min[sampled_dim]
        self.local_hidden_states_max = self.hidden_states_max[sampled_dim]
        self.local_max_min_gap = self.max_min_gap[sampled_dim]
        # self.bitwidth = torch.ceil(self.local_max_min_gap[-1] / (delta_x * 2))
        self.local_point = (self.local_hidden_states_min + self.local_hidden_states_max) / 2
        # self.local_point = torch.zeros_like(self.local_hidden_states_max)
        self.perturb_generation(select_dim)

        # print(self.hidden_states_mean[sampled_dim])
        # print(self.local_point)
        # print("layer {}, Local point {}".format(layer_index, self.local_point))

        self.ffn_gate_proj_weight = self.gate_proj.weight.data[ffn_dim, :]
        # self.ffn_up_proj_bias = self.up_proj.bias.data[ffn_dim]
        self.ffn_down_proj_weight = self.down_proj.weight.data[:, ffn_dim]
        self.ffn_up_proj_weight = self.up_proj.weight.data[ffn_dim, :]

        self.gate_proj_weight = self.gate_proj.weight.data[sampled_dim, :]
        # self.up_proj_bias = self.up_proj.bias.data[sampled_dim]
        self.down_proj_weight = self.down_proj.weight.data[:, sampled_dim]
        self.up_proj_weight = self.up_proj.weight.data[sampled_dim, :]
        self.dtype = self.gate_proj_weight.dtype

        # self.down_proj_bias = self.down_proj.bias.data
        # print("layer {}, Intermedia bias {}".format(layer_index, self.up_proj_bias))
        # ipdb.set_trace()

        self.local_approx_output = (self.act_fn(self.local_point).unsqueeze(0) * self.down_proj_weight)  # 4096, 512
        self.local_approx_output = self.local_approx_output
        self.grad_order = grad_order
        self.delta_hidden_state_thd = delta_hidden_state_thd
        self.grad_order_min = grad_order_min

        self.grad_update()
        # grad_matrix, _ = self.act_grad(self.local_point, grad_order, dtype=self.down_proj_weight.dtype)
        # self.discount_factor = self.discount_matrix()
        # self.grad_order_buf = torch.arange(1, self.grad_order + 1, device=self.ffn_up_proj_weight.device)
        # self.fuse_weight = self.down_proj_weight.unsqueeze(-1) * grad_matrix.unsqueeze(0) # 4096, 512, 12
        # self.grad_matrix = grad_matrix
        # print(self.down_proj_weight.device, grad_matrix.device, self.discount_factor.device)

        # encript the local point
        self.original_gate_proj_weight = self.gate_proj_weight.clone()
        self.original_down_proj_weight = self.down_proj_weight.clone()
        self.weight_perturb()
        self.dtype_trans_fn = self.to_fp32
        self.dtype_inverse_fn = self.to_fp16 if self.dtype == torch.float16 else self.to_bf16

        # self.local_approx_output = self.local_approx_output.type(self.dtype)
        # self.local_point = self.local_point.type(self.dtype)
        # self.fuse_weight = self.fuse_weight.type(self.dtype)

        # ipdb.set_trace()

        # self.free_local_weigth()

        del self.grad_matrix
        del self.local_hidden_states_min, self.local_hidden_states_max, self.local_max_min_gap
        del self.sampled_dim, self.ffn_dim, self.max_min_gap
        del self.up_proj, self.down_proj
        del self.down_proj_weight
        del self.hidden_states_mean, self.hidden_states_std, self.hidden_states_max, self.hidden_states_min
        del self.perturb_matrix
        del self.original_gate_proj_weight, self.original_down_proj_weight

        # self.fuse_weight = self.fuse_weight.cpu()

        # ipdb.set_trace()

    def act_grad(self, x, grad_order, dtype=torch.bfloat16):

        return silu_grad(x, grad_order, dtype)

    def discount_matrix(self):

        return torch.tensor(
            [torch.log(torch.arange(1, m + 1, device=self.up_proj_weight.device)).sum() \
             for m in range(1, self.grad_order + 1)], device=self.up_proj_weight.device).type(self.up_proj_weight.dtype)

    def grad_update(self):

        if self.grad_order > 0:
            grad_matrix, _ = self.act_grad(self.local_point, self.grad_order, dtype=self.down_proj_weight.dtype)
            self.discount_factor = self.discount_matrix()
            self.grad_order_buf = torch.arange(1, self.grad_order + 1, device=self.ffn_up_proj_weight.device)

            # mask = torch.tensor([True if idx == 0 or idx % 2 == 1 else False for idx in range(self.grad_order)])
            # grad_matrix = grad_matrix[:, mask]
            # self.discount_factor = self.discount_factor[mask]
            # self.grad_order_buf = self.grad_order_buf[mask]

            self.fuse_weight = self.down_proj_weight.unsqueeze(-1) * grad_matrix.unsqueeze(0)  # 4096, 512, 12
            self.grad_matrix = grad_matrix

        else:
            self.discount_factor = None
            self.grad_order_buf = None
            self.fuse_weight = None
            self.grad_matrix = None

    # def perturb_generation(self, select_dim, low=0.1, high=10):
    #     self.perturb_matrix = torch.rand(size=(select_dim,), device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype) * (high - low) + low
    #     self.perturb_matrix = self.perturb_matrix * (torch.randint(low=0, high=2, size=(select_dim,), device=self.up_proj.weight.device) * 2 - 1)

    def perturb_generation(self, select_dim, low=0.1, high=10):
        # self.perturb_matrix = torch.tensor([-1]*select_dim, device=self.up_proj.weight.device)
        self.perturb_matrix = torch.randint(low=0, high=2, size=(select_dim,),
                                            device=self.up_proj.weight.device) * 2 - 1

    def perturb_reverse(self):
        return torch.cat(
            [self.perturb_matrix.unsqueeze(-1) ** (tmp_grad_order) for tmp_grad_order in self.grad_order_buf], dim=-1)

    def weight_perturb(self):

        if self.grad_order > 0:
            self.gate_proj_weight = self.gate_proj_weight * self.perturb_matrix.unsqueeze(-1)
            self.local_point = self.local_point * self.perturb_matrix
            self.fuse_weight = self.fuse_weight * self.perturb_reverse()

    def load_local_weight(self, device):
        self.fuse_weight = self.fuse_weight.to(device)
        self.local_approx_output = self.local_approx_output.to(device)

    def free_local_weigth(self):
        self.fuse_weight = self.fuse_weight.cpu()
        self.local_approx_output = self.local_approx_output.cpu()

    def to_fp32(self, *argv):
        self.up_proj_weight = self.up_proj_weight.type(torch.float32)
        self.gate_proj_weight = self.gate_proj_weight.type(torch.float32)
        self.local_point = self.local_point.type(torch.float32)
        self.local_approx_output = self.local_approx_output.type(torch.float32)
        self.discount_factor = self.discount_factor.type(torch.float32)
        self.fuse_weight = self.fuse_weight.type(torch.float32)
        self.up_proj_weight = self.up_proj_weight.type(torch.float32)
        # self.original_gate_proj_weight = self.original_gate_proj_weight.type(torch.float32)
        # self.original_down_proj_weight = self.original_down_proj_weight.type(torch.float32)
        return (arg.type(torch.float32) for arg in argv)

    def to_fp16(self, *argv):
        self.up_proj_weight = self.up_proj_weight.type(torch.float16)
        self.gate_proj_weight = self.gate_proj_weight.type(torch.float16)
        self.local_point = self.local_point.type(torch.float16)
        self.local_approx_output = self.local_approx_output.type(torch.float16)
        self.discount_factor = self.discount_factor.type(torch.float16)
        self.fuse_weight = self.fuse_weight.type(torch.float16)
        self.up_proj_weight = self.up_proj_weight.type(torch.float16)
        # self.original_gate_proj_weight = self.original_gate_proj_weight.type(torch.float16)
        # self.original_down_proj_weight = self.original_down_proj_weight.type(torch.float16)
        return (arg.type(torch.float16) for arg in argv)

    def to_bf16(self, *argv):
        self.up_proj_weight = self.up_proj_weight.type(torch.bfloat16)
        self.gate_proj_weight = self.gate_proj_weight.type(torch.bfloat16)
        self.local_point = self.local_point.type(torch.bfloat16)
        self.local_approx_output = self.local_approx_output.type(torch.bfloat16)
        self.discount_factor = self.discount_factor.type(torch.bfloat16)
        self.fuse_weight = self.fuse_weight.type(torch.bfloat16)
        self.up_proj_weight = self.up_proj_weight.type(torch.bfloat16)
        # self.original_gate_proj_weight = self.original_gate_proj_weight.type(torch.bfloat16)
        # self.original_down_proj_weight = self.original_down_proj_weight.type(torch.bfloat16)
        return (arg.type(torch.bfloat16) for arg in argv)

    def approx_output_parallel(self, hidden_states):

        approx_up_output = (hidden_states @ self.up_proj_weight.T).unsqueeze(-2)
        hidden_states_ffn1 = (hidden_states @ self.gate_proj_weight.T)

        delta_hidden_state = hidden_states_ffn1 - self.local_point
        delta_hidden_state_neg_index = (delta_hidden_state > 0) * 2 - 1
        delta_hidden_state = delta_hidden_state.abs()
        log_discount_term = self.grad_order_buf.unsqueeze(0).unsqueeze(0).unsqueeze(0) * torch.log(
            delta_hidden_state).unsqueeze(-1) - self.discount_factor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        discount_term = torch.exp(log_discount_term)
        odd_grad_order_index = (self.grad_order_buf % 2).type(torch.bool)
        discount_term_odd_order = discount_term[:, :, :, odd_grad_order_index]
        # ipdb.set_trace()
        discount_term[:, :, :, odd_grad_order_index] = discount_term_odd_order * delta_hidden_state_neg_index.unsqueeze(
            -1)
        approx_gate_output = torch.sum(self.fuse_weight * discount_term.unsqueeze(-3), dim=(-1))
        # ipdb.set_trace()
        approx_output = ((approx_gate_output + self.local_approx_output) * approx_up_output).sum(-1)

        return approx_output

    @torch.no_grad()
    def approx_output_log_exp(self, hidden_states):

        # print(f"layer index: {self.layer_index}")
        # print(f"Hidden state shape: {hidden_states.shape}")

        approx_up_output = (hidden_states @ self.up_proj_weight.T)  # .unsqueeze(-1)
        hidden_states_ffn1 = (hidden_states @ self.gate_proj_weight.T)

        delta_hidden_state = hidden_states_ffn1 - self.local_point
        delta_hidden_state_neg_factor = (delta_hidden_state > 0).type(torch.int8) * 2 - 1
        delta_hidden_state_abs = delta_hidden_state.abs()
        log_delta_hidden_state = torch.log(delta_hidden_state_abs)
        # del hidden_states_ffn1, delta_hidden_state

        over_act_mask = (delta_hidden_state_abs > self.delta_hidden_state_thd).type(torch.int8)

        approx_output = approx_up_output @ self.local_approx_output.T
        if self.grad_order == 0:
            return approx_output

        for idx, tmp_grad_order in enumerate(self.grad_order_buf):
            log_discount_term = log_delta_hidden_state * tmp_grad_order - self.discount_factor[idx]
            odd_grad_order_flag = (tmp_grad_order % 2).type(torch.bool)
            if odd_grad_order_flag:
                discount_term = torch.exp(log_discount_term) * delta_hidden_state_neg_factor
            else:
                discount_term = torch.exp(log_discount_term)

            # del log_discount_term
            if tmp_grad_order <= self.grad_order_min:
                tmp_value = discount_term * approx_up_output
            else:
                tmp_value = discount_term * approx_up_output * (1 - over_act_mask)
            # del discount_term

            term_value = tmp_value @ self.fuse_weight[:, :, idx].T
            # del tmp_value

            approx_output = approx_output + term_value
            # del term_value

        # ipdb.set_trace()
        return approx_output

    @torch.no_grad()
    def approx_output_power(self, hidden_states):

        approx_up_output = (hidden_states @ self.up_proj_weight.T)  # .unsqueeze(-1)
        hidden_states_ffn1 = (hidden_states @ self.gate_proj_weight.T)

        delta_hidden_state = hidden_states_ffn1 - self.local_point
        del hidden_states_ffn1

        approx_output = approx_up_output @ self.local_approx_output.T
        if self.grad_order == 0:
            return approx_output

        for idx, tmp_grad_order in enumerate(self.grad_order_buf):
            power_term = torch.ones_like(delta_hidden_state, dtype=delta_hidden_state.dtype,
                                         device=delta_hidden_state.device)
            for power_idx in range(tmp_grad_order):
                power_term = power_term * (delta_hidden_state / power_idx)

            tmp_value = power_term * approx_up_output
            del power_term

            term_value = tmp_value @ self.fuse_weight[:, :, idx].T
            del tmp_value

            approx_output = approx_output + term_value
            del term_value

        return approx_output

    def ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        up_output = hidden_states @ self.ffn_up_proj_weight.T
        gate_output = hidden_states @ self.ffn_gate_proj_weight.T
        ffn_output = (self.act_fn(gate_output) * up_output) @ self.ffn_down_proj_weight.T

        return ffn_output

    def forward_log_exp(self, hidden_states: torch.Tensor) -> torch.Tensor:

        approx_output = self.approx_output_log_exp(hidden_states)
        ffn_output = self.ffn_forward(hidden_states)
        hidden_states_total = approx_output + ffn_output

        return hidden_states_total

    def forward_power(self, hidden_states: torch.Tensor) -> torch.Tensor:

        approx_output = self.approx_output_power(hidden_states)
        ffn_output = self.ffn_forward(hidden_states)
        hidden_states_total = approx_output + ffn_output

        return hidden_states_total

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_log_exp(hidden_states)

    # def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #
    #     # self.fuse_weight = self.fuse_weight.to(self.ffn_up_proj_weight.device)
    #
    #     # hidden_states, = self.dtype_trans_fn(hidden_states)
    #     # print(hidden_states.shape)
    #     approx_output = self.approx_output_serial(hidden_states)
    #     # approx_output_std = self.exact_forward(hidden_states)
    #     # approx_output__ = self.approx_output_parallel(hidden_states)
    #
    #     # print((approx_output - approx_output_std).abs().sum())
    #     # print((approx_output - approx_output__).abs().sum())
    #     # ipdb.set_trace()
    #
    #     # approx_output, hidden_states = self.dtype_inverse_fn(approx_output, hidden_states)
    #
    #     # hidden_states_ffn1 = (hidden_states @ self.gate_proj_weight.T)
    #     # hidden_error = (hidden_states_ffn1 - self.local_point).abs()
    #     # print(hidden_error.max())
    #
    #
    #
    #     hidden_states_total = approx_output + ffn_output
    #
    #     # print(approx_output_)
    #     # # print(approx_output__)
    #     # print(approx_output)
    #     # print((approx_output - approx_output_std).abs().sum())
    #     # self.fuse_weight = self.fuse_weight.cpu()
    #
    #     return hidden_states_total

    def exact_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        up_output = hidden_states @ self.up_proj_weight.T
        gate_output = hidden_states @ self.original_gate_proj_weight.T
        return (self.act_fn(gate_output) * up_output) @ self.original_down_proj_weight.T


# class Tylor_expansion_LlamaDecoderLayer(nn.Module):
#     """This corresponds to the Block class in the timm implementation."""
#
#     def __init__(self, llama_layer) -> None:
#         super().__init__()
#
#         self.hidden_size = llama_layer.hidden_size
#         self.self_attn = llama_layer.self_attn
#         self.mlp = llama_layer.mlp
#         self.input_layernorm = llama_layer.input_layernorm
#         self.post_attention_layernorm = llama_layer.post_attention_layernorm

def hiddenstates_benchmark(model, fname):
    return base_hiddenstates_benchmark(model, fname)


def model_debug(model):
    return base_model_debug(model, LlamaMLP_)


def model_tylor_expansion(model,
                           param_fname=None,
                           expand_layer=None,
                           select_dim=256,
                           grad_order=12,
                           max_gap=0,
                           grad_order_min=4,
                           delta_hidden_state_thd=2.5):
    return base_model_tylor_expansion(model,
                                      Tylor_expansion_LlamaMLP,
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

# def approx_output_serial(self, hidden_states):
#
#     approx_up_output = (hidden_states @ self.up_proj_weight.T).unsqueeze(-1)
#     hidden_states_ffn1 = (hidden_states @ self.gate_proj_weight.T)
#
#     delta_hidden_state = hidden_states_ffn1 - self.local_point
#     delta_hidden_state_neg_factor = (delta_hidden_state > 0) * 2 - 1
#     delta_hidden_state = delta_hidden_state.abs()
#
#     hidden_states_ffn1_2 = (hidden_states @ self.gate_proj_weight2.T)
#     delta_hidden_state2 = hidden_states_ffn1_2 - self.local_point2
#     delta_hidden_state_neg_factor2 = (delta_hidden_state2 > 0) * 2 - 1
#     delta_hidden_state2 = delta_hidden_state2.abs()
#     approx_output2 = self.local_approx_output.unsqueeze(0).unsqueeze(0) @ approx_up_output
#
#     approx_output = self.local_approx_output.unsqueeze(0).unsqueeze(0) @ approx_up_output
#     for grad_idx in range(self.grad_order):
#         log_discount_term = torch.log(delta_hidden_state) * self.grad_order_buf[grad_idx] - self.discount_factor[grad_idx]
#         odd_grad_order_flag = (self.grad_order_buf[grad_idx] % 2).type(torch.bool)
#         if odd_grad_order_flag:
#             discount_term = torch.exp(log_discount_term) * delta_hidden_state_neg_factor
#         else:
#             discount_term = torch.exp(log_discount_term)
#
#         # term_value = self.fuse_weight[:, :, grad_idx].unsqueeze(0).unsqueeze(0) * discount_term.unsqueeze(-2)
#         # approx_output = approx_output + (term_value @ approx_up_output)
#
#         tmp_value = discount_term.unsqueeze(-1) * approx_up_output
#         term_value = self.fuse_weight[:, :, grad_idx].unsqueeze(0).unsqueeze(0) @ tmp_value
#         approx_output = approx_output + term_value
#
#
#         log_discount_term2 = torch.log(delta_hidden_state2) * self.grad_order_buf[grad_idx] - self.discount_factor[grad_idx]
#         if odd_grad_order_flag:
#             discount_term2 = torch.exp(log_discount_term2) * delta_hidden_state_neg_factor2
#         else:
#             discount_term2 = torch.exp(log_discount_term2)
#
#         tmp_value2 = discount_term2.unsqueeze(-1) * approx_up_output
#         term_value2 = self.fuse_weight2[:, :, grad_idx].unsqueeze(0).unsqueeze(0) @ tmp_value2
#         approx_output2 = approx_output2 + term_value2
#         print(term_value[0,0,:10,0])
#         print(term_value2[0,0,:10,0])
#
#
#         # ipdb.set_trace()
#         # CyxA
#
#         # del tmp_value, term_value
#     print(approx_output[0,0,:10,0])
#     print(approx_output2[0,0,:10,0])
#     ipdb.set_trace()
#
#     return approx_output.squeeze(-1)