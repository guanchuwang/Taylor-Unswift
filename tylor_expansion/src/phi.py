from typing import Dict, List, Optional, Set, Tuple, Union
import torch
from torch import nn
import ipdb, re
# from transformers.models.phi.modeling_phi import PhiMLP, PhiDecoderLayer
from .utils import gelu_grad
from .utils import model_debug as base_model_debug
from .utils import model_tylor_expansion as base_model_tylor_expansion
from .utils import hiddenstates_benchmark as base_hiddenstates_benchmark
from transformers.activations import GELUActivation


class PhiMLP_(nn.Module):

    def __init__(self, phi_mlp, config, layer_index=0) -> None:
        super().__init__()

        self.config = config
        self.fc1 = phi_mlp.fc1
        self.fc2 = phi_mlp.fc2
        self.act_fn = GELUActivation()  # phi_mlp.activation_fn

        self.intermediate_size, self.hidden_size = phi_mlp.fc1.weight.data.shape
        self.hidden_states_intermediate = None  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # None #
        self.hidden_states_intermediate_mean = 0  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # 0
        self.hidden_states_intermediate_square_mean = 0  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # 0
        self.hidden_states_intermediate_std = 0  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # 0
        self.hidden_states_intermediate_max = None  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # None
        self.hidden_states_intermediate_min = None  # torch.zeros(self.intermediate_size, device=phi_mlp.fc1.weight.device) # None
        self.batch_num = 0
        self.layer_index = layer_index
        print(layer_index)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        self.batch_num = self.batch_num + 1
        print(self.batch_num)

        hidden_states_intermediate = (hidden_states @ self.fc1.weight.T)
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

        # if self.batch_num == 17:
        #     # print(hidden_states_intermediate[:, 1128])
        #     # print(self.hidden_states_intermediate_mean[1128])
        #     ipdb.set_trace()

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

        output = self.fc2(self.act_fn(self.fc1(hidden_states)))

        # ipdb.set_trace()
        # print(self.layer_index)

        return output


class Tylor_expansion_PhiMLP(nn.Module):
    def __init__(self, phi_mlp, config, grad_order=6, select_dim=4, max_gap=10,
                 hidden_states_mean=None, hidden_states_std=None, hidden_states_max=None,
                 hidden_states_min=None, grad_order_min=4, delta_hidden_state_thd=2, bitwidth=1, layer_index=0) -> None:
        super().__init__()

        self.config = config
        self.fc2 = phi_mlp.fc2
        self.fc1 = phi_mlp.fc1
        self.act_fn = GELUActivation()  # phi_mlp.activation_fn
        self.layer_index = layer_index

        self.intermediate_size, self.hidden_size = phi_mlp.fc1.weight.data.shape
        self.hidden_states_mean = hidden_states_mean.to(self.fc1.weight.device)
        self.hidden_states_std = hidden_states_std.to(self.fc1.weight.device)
        self.hidden_states_max = hidden_states_max.to(self.fc1.weight.device)
        self.hidden_states_min = hidden_states_min.to(self.fc1.weight.device)
        self.max_min_gap = self.hidden_states_max - self.hidden_states_min

        # self.max_min_gap = torch.randn(self.intermediate_size)

        std_sort_index = self.max_min_gap.argsort()
        sampled_dim = std_sort_index[:select_dim]
        ffn_dim = std_sort_index[select_dim:]
        self.sampled_dim = sampled_dim
        self.ffn_dim = ffn_dim
        print("layer {}, Max min gap {}, {}, {}".format(layer_index, self.max_min_gap[self.sampled_dim].min(),
                                                        self.max_min_gap[self.sampled_dim].median(),
                                                        self.max_min_gap[self.sampled_dim].max()))

        self.local_hidden_states_min = self.hidden_states_min[sampled_dim]
        self.local_hidden_states_max = self.hidden_states_max[sampled_dim]
        self.local_max_min_gap = self.max_min_gap[sampled_dim]
        self.local_point = (self.local_hidden_states_min + self.local_hidden_states_max) / 2

        # print(self.hidden_states_mean[sampled_dim])
        # print(self.local_point)
        # print("layer {}, Local point {}".format(layer_index, self.local_point))

        self.ffn_fc1_weight = self.fc1.weight.data[ffn_dim, :]
        self.ffn_fc1_bias = self.fc1.bias.data[ffn_dim]
        self.ffn_fc2_weight = self.fc2.weight.data[:, ffn_dim]
        self.dtype = self.ffn_fc2_weight.dtype

        self.fc1_weight = self.fc1.weight.data[sampled_dim, :]
        self.fc1_bias = self.fc1.bias.data[sampled_dim]
        self.fc2_weight = self.fc2.weight.data[:, sampled_dim]
        self.fc2_bias = self.fc2.bias.data

        # self.fc2_bias = self.fc2.bias.data
        # print("layer {}, Intermedia bias {}".format(layer_index, self.fc1_bias))
        # ipdb.set_trace()

        self.local_approx_output = (
                    self.act_fn(self.local_point + self.fc1_bias).unsqueeze(0) @ self.fc2_weight.T)  # 4096, 512
        self.local_approx_output_and_bias = self.local_approx_output + self.fc2_bias
        self.grad_order = grad_order
        self.grad_update()

        self.delta_hidden_state_thd = delta_hidden_state_thd
        self.grad_order_min = grad_order_min

        # self.local_approx_output = self.local_approx_output.type(self.dtype)
        # self.local_point = self.local_point.type(self.dtype)
        # self.fuse_weight = self.fuse_weight.type(self.dtype)

        # del self.grad_matrix, self.fc2_weight
        # del self.local_hidden_states_min, self.local_hidden_states_max, self.local_max_min_gap
        # del self.sampled_dim, self.ffn_dim, self.max_min_gap
        # del self.fc1, self.fc2, self.layer_index
        # del self.hidden_states_mean, self.hidden_states_std, self.hidden_states_max, self.hidden_states_min

        # ipdb.set_trace()

    def act_grad(self, x, grad_order, dtype=torch.bfloat16):

        return gelu_grad(x, grad_order, dtype=dtype)

    def discount_matrix(self):

        return torch.tensor(
            [torch.log(torch.arange(1, m + 1, device=self.fc1_weight.device)).sum() \
             for m in range(1, self.grad_order + 1)], device=self.fc1_weight.device).type(self.dtype)

    def grad_update(self):

        if self.grad_order > 0:
            grad_matrix, _ = self.act_grad(self.local_point, self.grad_order, dtype=self.dtype)
            self.discount_factor = self.discount_matrix()
            self.grad_order_buf = torch.arange(1, self.grad_order + 1, device=self.ffn_fc1_weight.device)
            # print(self.fc2_weight.device, grad_matrix.device, self.discount_factor.device)

            self.fuse_weight = self.fc2_weight.unsqueeze(-1) * grad_matrix.unsqueeze(0)  # 4096, 512, 12
            self.grad_matrix = grad_matrix

        else:
            self.discount_factor = None
            self.grad_order_buf = None
            self.fuse_weight = None
            self.grad_matrix = None

    def approx_output_parallel(self, hidden_states):

        hidden_states_ffn1 = (hidden_states @ self.fc1_weight.T)
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
        approx_output = torch.sum(self.fuse_weight * discount_term.unsqueeze(-3), dim=(-1, -2))
        approx_output = approx_output + self.local_approx_output_and_bias
        return approx_output

    @torch.no_grad()
    def approx_output_log_exp(self, hidden_states):

        # print(f"layer index: {self.layer_index}")
        # print(f"Hidden state shape: {hidden_states.shape}"
        hidden_states_ffn1 = (hidden_states @ self.fc1_weight.T)

        delta_hidden_state = hidden_states_ffn1 - self.local_point
        delta_hidden_state_neg_factor = (delta_hidden_state > 0).type(torch.int8) * 2 - 1
        delta_hidden_state_abs = delta_hidden_state.abs()
        log_delta_hidden_state = torch.log(delta_hidden_state_abs)
        over_act_mask = (delta_hidden_state_abs > self.delta_hidden_state_thd).type(torch.int8)

        approx_output = self.local_approx_output_and_bias
        if self.grad_order == 0:
            return approx_output

        for idx, tmp_grad_order in enumerate(self.grad_order_buf):
            log_discount_term = log_delta_hidden_state * tmp_grad_order - self.discount_factor[idx]
            odd_grad_order_flag = (tmp_grad_order % 2).type(torch.bool)
            if odd_grad_order_flag:
                discount_term = torch.exp(log_discount_term) * delta_hidden_state_neg_factor
            else:
                discount_term = torch.exp(log_discount_term)

            if tmp_grad_order <= self.grad_order_min:
                term_value = discount_term @ self.fuse_weight[:, :, idx].T
            else:
                term_value = (discount_term * (1 - over_act_mask)) @ self.fuse_weight[:, :, idx].T

            approx_output = approx_output + term_value

            # print(term_value)
        # ipdb.set_trace()

        return approx_output

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        approx_output = self.approx_output_log_exp(hidden_states)
        # approx_output = self.approx_output_parallel(hidden_states)

        # hidden_states_ffn1 = (hidden_states @ self.fc1_weight.T)
        # hidden_error = (hidden_states_ffn1 - self.local_point).abs()
        ffn_output = self.act_fn(hidden_states @ self.ffn_fc1_weight.T + \
                                 self.ffn_fc1_bias) @ self.ffn_fc2_weight.T

        hidden_states_total = approx_output + ffn_output

        # print(hidden_states_total.shape)
        # exact_output = self.act_fn(hidden_states @ self.fc1_weight.T + \
        #  self.fc1_bias) @ self.fc2_weight.T + self.fc2_bias

        # print(approx_output)
        # print(approx_output_parallel)
        # print(exact_output)
        # ipdb.set_trace()

        # gate_output = self.fc1(hidden_states)
        # up_output = self.fc1(hidden_states)
        # output = self.fc2(self.act_fn(gate_output) * up_output)
        # print(output[0,0,:10])
        # print(hidden_states_total[0,0,:10])
        # ipdb.set_trace()

        return hidden_states_total


def hiddenstates_benchmark(model, fname):
    return base_hiddenstates_benchmark(model, fname)


def model_debug(model):
    return base_model_debug(model, PhiMLP_)


def model_tylor_expansion(model,
                          param_fname=None,
                          expand_layer=None,
                          select_dim=256,
                          grad_order=12,
                          max_gap=0,
                          grad_order_min=4,
                          delta_hidden_state_thd=2):
    return base_model_tylor_expansion(model,
                                      Tylor_expansion_PhiMLP,
                                      param_fname,
                                      expand_layer,
                                      select_dim=select_dim,
                                      grad_order=grad_order,
                                      max_gap=max_gap,
                                      grad_order_min=grad_order_min,
                                      delta_hidden_state_thd=delta_hidden_state_thd)


