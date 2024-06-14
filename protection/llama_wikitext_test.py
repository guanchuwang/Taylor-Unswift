
import torch
from torch import nn

from datasets import load_dataset
import ipdb, copy, sys, os
import numpy as np
import time
import json

sys.path.append("../")
sys.path.append("./")

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from tylor_expansion.llama import model_tylor_expansion
# from util import model_expand, print_model_parameters, freeze_model_parameters, zero_model_parameters, ViTBase2Large

os.environ["CUDA_VISIBLE_DEVICES"]="0"
dtype = "torch.bfloat16"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", cache_dir="/scratch")

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             cache_dir="/scratch",
                                             torch_dtype=eval(dtype),
                                             use_flash_attention_2=True
                                             )
model_expansion = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",
                                             cache_dir="/scratch",
                                             torch_dtype=eval(dtype), # torch.float32, # torch.float16 #
                                             use_flash_attention_2=True
                                             )
# context_buf = [ "Robert <unk> is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John <unk> in 2002 . In 2004 <unk> landed a role as ' Craig ' in the episode ' Teddy 's Story ' of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the <unk> <unk> Factory in London . He was directed by John <unk> and starred alongside Ben <unk> , Shane <unk> , Harry Kent , Fraser <unk> , Sophie Stanton and Dominic Hall ."]
context_buf = [ "Although initially he was little @-@ known to other writers , his works came to be hugely influential in both Chinese and Japanese literary culture . Of his poetic writing , nearly fifteen hundred poems have been preserved over the ages . He has been called the ' Poet @-@ Historian ' and the ' Poet @-@ Sage ' by Chinese critics , while the range of his work has allowed him to be introduced to Western readers as ' the Chinese Virgil , Horace , <unk> , Shakespeare , Milton , Burns , <unk> , <unk> , Hugo or <unk> ' ."]

context = " ".join(context_buf)

prompt = [
  {"role": "user", "content": context}
]

# prompt = [
#     {"role": "user", "content": "Tell me a story."}
# ]

prompt_adapt = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt_adapt, return_tensors="pt")
inputs.input_ids = inputs.input_ids.cuda()

model = model.cuda()
generate_ids1 = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=False)
outputs1 = tokenizer.batch_decode(generate_ids1[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(f"Original output: {outputs1}")
# print("============================")
del model

select_dim = 8000
grad_order = 8
grad_order_min = 8
delta_hidden_state_thd = 2.5
expand_layer = None

model_expansion = model_expansion.cuda()
model_tylor_expansion(model_expansion,
                      "./output/llama-3-8b-hf-pileval-hidden-states/hidden-states-" + dtype + "-n-20000-len-4096.pth.tar",
                      select_dim=select_dim,
                      grad_order=grad_order,
                      expand_layer=None,
                      grad_order_min=grad_order_min,
                      delta_hidden_state_thd=delta_hidden_state_thd)
generate_ids2 = model_expansion.generate(inputs.input_ids, max_new_tokens=256, do_sample=False) # do_sample=True, top_p=0.9) #
outputs2 = tokenizer.batch_decode(generate_ids2[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(f"After protection: {outputs2}")
# print("============================")
print(f"Original output: {outputs1}")
print("============================")
print(f"After protection: {outputs2}")
print("============================")

model_tylor_log = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "dtype": dtype,
    "context": context,
    "original_output": outputs1,
    "llm_protection_conifg":
    {
        "select_dim": select_dim,
        "grad_order": grad_order,
        "expand_layer": expand_layer,
        "grad_order_min": grad_order_min,
        "delta_hidden_state_thd": delta_hidden_state_thd,
    },
    "llm_protection_output": outputs2,
}

with open(f"./case_study/wikitext-2/wikitext-2_protection_W_{select_dim}_N_{grad_order}.json", "w") as f:
    json.dump(model_tylor_log, f, indent=4)

# ipdb.set_trace()
