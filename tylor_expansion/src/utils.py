import torch
import math
from scipy.special import comb
import ipdb


def gaussian_cdf(x):
    return (1 + torch.erf(x/math.sqrt(2))) * 0.5
    # return (torch.nn.functional.gelu(x) + 1e-20) / (x + 1e-20)

def gaussian_pdf(x):
    return 1/math.sqrt(2 * torch.pi) * torch.exp(- x**2 / 2)


def gelu_grad(x, grad_order, dtype=torch.bfloat16):
    y__n_buf = torch.zeros(len(x), grad_order, device=x.device, dtype=dtype)
    gelu_grad_buf = torch.zeros(len(x), grad_order, device=x.device, dtype=dtype)
    y__n_buf_0 = gaussian_pdf(x)
    y__n_buf[:, 0] = (-x) * gaussian_pdf(x)
    gelu_grad_buf[:, 0] = gaussian_cdf(x) + x * gaussian_pdf(x)
    # gelu_grad_buf[:, 1] = x * x * gaussian_pdf(x) - gaussian_pdf(x)
    for grad_idx in range(1, grad_order):
        if grad_idx == 1:  # second order gradient
            y__n_buf[:, grad_idx] = (-x) * y__n_buf[:, 0] - y__n_buf_0
            gelu_grad_buf[:, grad_idx] = x * y__n_buf[:, 0] + (grad_idx + 1) * y__n_buf_0
        else:  # grad_idx+1 order gradient
            y__n_buf[:, grad_idx] = (-x) * y__n_buf[:, grad_idx - 1] - grad_idx * y__n_buf[:, grad_idx - 2]
            gelu_grad_buf[:, grad_idx] = x * y__n_buf[:, grad_idx - 1] + (grad_idx + 1) * y__n_buf[:, grad_idx - 2]

    return gelu_grad_buf, y__n_buf

def silu_grad(x, grad_order, dtype=torch.bfloat16): #

    y__n_buf = torch.zeros(len(x), grad_order, device=x.device, dtype=dtype)
    silu_grad_buf = torch.zeros(len(x), grad_order, device=x.device, dtype=dtype)
    y__n_buf_0 = torch.nn.functional.sigmoid(x)
    y__n_buf[:, 0] = torch.nn.functional.sigmoid(x) * (1 - torch.nn.functional.sigmoid(x))
    silu_grad_buf[:, 0] = x * y__n_buf[:, 0] + 1 * y__n_buf_0
    # silu_grad_buf[:, 1] = x * x * gaussian_pdf(x) - gaussian_pdf(x)
    for grad_idx in range(1, grad_order):
        if grad_idx == 1: # second order gradient
            y__n_buf[:, grad_idx] = y__n_buf[:, 0] - 2 * y__n_buf_0 * y__n_buf[:, 0]
        else: # grad_idx+1 order gradient
            for idx in range(grad_idx + 1): # k = 0, ..., n
                coeff = comb(grad_idx, idx)
                if idx == 0:
                    grad_term = coeff * (-1) * y__n_buf_0 * y__n_buf[:, grad_idx-1]
                elif idx == grad_idx:
                    grad_term = coeff * (y__n_buf[:, grad_idx-1] - y__n_buf[:, grad_idx-1] * y__n_buf_0)
                else:
                    grad_term = coeff * (-1) * y__n_buf[:, idx-1] * y__n_buf[:, grad_idx-idx-1]

                y__n_buf[:, grad_idx] = y__n_buf[:, grad_idx] + grad_term

        silu_grad_buf[:, grad_idx] = x * y__n_buf[:, grad_idx] + (grad_idx+1) * y__n_buf[:, grad_idx-1]

    # print(silu_grad_buf.shape)
    return silu_grad_buf.detach(), y__n_buf.detach()


def min_dim_selection(max_min_gap, select_dim_num):
    std_sort_index = max_min_gap.argsort()
    approx_dim = std_sort_index[:select_dim_num]
    ffn_dim = std_sort_index[select_dim_num:]
    return approx_dim, ffn_dim

def dim_selection(max_min_gap, select_dim_num, max_gap=15):
    max_min_gap_sort, std_sort_index = torch.sort(max_min_gap)
    mask = (max_min_gap_sort < max_gap)
    if mask.sum() >= select_dim_num:
        valid_index = std_sort_index[mask]
        approx_dim = valid_index[-select_dim_num:]
        ffn_dim = torch.cat([valid_index[0:-select_dim_num], std_sort_index[~mask]], dim=0)

    else:
        approx_dim = std_sort_index[:select_dim_num]
        ffn_dim = std_sort_index[select_dim_num:]
    # ffn_dim = torch.tensor(list(set(std_sort_index) - set(approx_dim)), device=approx_dim.device)
    # print(max_min_gap[approx_dim])
    # ipdb.set_trace()

    return approx_dim, ffn_dim


def hiddenstates_benchmark(model, fname):

    hidden_states_buf = {}
    for layer_index in range(len(model.model.layers)):

        hidden_states_intermediate_mean = model.model.layers[layer_index].mlp.hidden_states_intermediate_mean.cpu()
        hidden_states_intermediate_std = model.model.layers[layer_index].mlp.hidden_states_intermediate_std.cpu()
        hidden_states_intermediate_max = model.model.layers[layer_index].mlp.hidden_states_intermediate_max.cpu()
        hidden_states_intermediate_min = model.model.layers[layer_index].mlp.hidden_states_intermediate_min.cpu()
        hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_mean".format(layer_index)] = hidden_states_intermediate_mean
        hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_std".format(layer_index)] = hidden_states_intermediate_std
        hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_max".format(layer_index)] = hidden_states_intermediate_max
        hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_min".format(layer_index)] = hidden_states_intermediate_min

    torch.save(hidden_states_buf, fname)


def model_debug(model, MLP_class):

    for layer_index in range(len(model.model.layers)):

        llama_mlp = model.model.layers[layer_index].mlp
        model.model.layers[layer_index].mlp = MLP_class(llama_mlp, model.config, layer_index)


def model_tylor_expansion(model,
                          MLP_class,
                          param_fname=None,
                          expand_layer=None,
                          select_dim=256,
                          grad_order=12,
                          max_gap=0,
                          grad_order_min=4,
                          delta_hidden_state_thd=2.5):

    # print(model, MLP_class, param_fname, expand_layer, select_dim, grad_order, max_gap)

    if str(model.dtype) not in param_fname:
        raise RuntimeError("You must give the dtype of hidden states value")

    hidden_states_buf = torch.load(param_fname)

    if expand_layer is None:
        expand_layer = torch.arange(len(model.model.layers))

    for layer_index in expand_layer:
    # for layer_index in [0]:

        llama_mlp = model.model.layers[layer_index].mlp

        hidden_states_mean = hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_mean".format(layer_index)]
        hidden_states_std = hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_std".format(layer_index)]
        hidden_states_max = hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_max".format(layer_index)]
        hidden_states_min = hidden_states_buf["model.layer[{}].mlp.hidden_states_intermediate_min".format(layer_index)]

        # hidden_states_mean = torch.zeros(llama_mlp.intermediate_size,)
        # hidden_states_std = torch.zeros(llama_mlp.intermediate_size,)
        # hidden_states_max = torch.zeros(llama_mlp.intermediate_size,)
        # hidden_states_min = torch.zeros(llama_mlp.intermediate_size,)
        # ipdb.set_trace()
        model.model.layers[layer_index].mlp = MLP_class(llama_mlp, model.config,
                select_dim=select_dim, grad_order=grad_order, max_gap=max_gap,
                hidden_states_mean=hidden_states_mean, hidden_states_std=hidden_states_std,
                hidden_states_max=hidden_states_max, hidden_states_min=hidden_states_min, layer_index=(-1 if layer_index == len(model.model.layers) - 1 else layer_index),
                grad_order_min=grad_order_min, delta_hidden_state_thd=delta_hidden_state_thd)
        # if layer_index == 5:
        #     break

        # print(f"Hiding Layer {layer_index} weight ...")

    # ipdb.set_trace()