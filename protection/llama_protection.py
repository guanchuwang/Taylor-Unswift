
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

os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
context_buf = [f"New York (CNN) -- More than 80 Michael Jackson collectibles -- including the late pop star's famous rhinestone-studded glove from a 1983 performance -- were auctioned off Saturday, reaping a total $2 million. Profits from the auction at the Hard Rock Cafe in New York's Times Square crushed pre-sale expectations of only $120,000 in sales. The highly prized memorabilia, which included items spanning the many stages of Jackson's career, came from more than 30 fans, associates and family members, who contacted Julien's Auctions to sell their gifts and mementos of the singer. Jackson's flashy glove was the big-ticket item of the night, fetching $420,000 from a buyer in Hong Kong, China. Jackson wore the glove at a 1983 performance during 'Motown 25,' an NBC special where he debuted his revolutionary moonwalk. Fellow Motown star Walter 'Clyde' Orange of the Commodores, who also performed in the special 26 years ago, said he asked for Jackson's autograph at the time, but Jackson gave him the glove instead. 'The legacy that [Jackson] left behind is bigger than life for me,' Orange said. 'I hope that through that glove people can see what he was trying to say in his music and what he said in his music.' Orange said he plans to give a portion of the proceeds to charity. Hoffman Ma, who bought the glove on behalf of Ponte 16 Resort in Macau, paid a 25 percent buyer's premium, which was tacked onto all final sales over $50,000. Winners of items less than $50,000 paid a 20 percent premium."]
question_buf = [ "Where was the Auction held?", "How much did they make?", "How much did they expected?", "WHo buy the Jackson Glove", "Where was the buyer of the glove from?" ]

# context_buf = [
#     "Chronic rhinosinusitis (CRS) is a heterogeneous disease with an uncertain pathogenesis. Group 2 innate lymphoid cells (ILC2s) represent a recently discovered cell population which has been implicated in driving Th2 inflammation in CRS; however, their relationship with clinical disease characteristics has yet to be investigated.",
#     "The aim of this study was to identify ILC2s in sinus mucosa in patients with CRS and controls and compare ILC2s across characteristics of disease.",
#     "A cross-sectional study of patients with CRS undergoing endoscopic sinus surgery was conducted. Sinus mucosal biopsies were obtained during surgery and control tissue from patients undergoing pituitary tumour resection through transphenoidal approach. ILC2s were identified as CD45(+) Lin(-) CD127(+) CD4(-) CD8(-) CRTH2(CD294)(+) CD161(+) cells in single cell suspensions through flow cytometry. ILC2 frequencies, measured as a percentage of CD45(+) cells, were compared across CRS phenotype, endotype, inflammatory CRS subtype and other disease characteristics including blood eosinophils, serum IgE, asthma status and nasal symptom score.",
#     "35 patients (40% female, age 48 +/- 17 years) including 13 with eosinophilic CRS (eCRS), 13 with non-eCRS and 9 controls were recruited. ILC2 frequencies were associated with the presence of nasal polyps (P = 0.002) as well as high tissue eosinophilia (P = 0.004) and eosinophil-dominant CRS (P = 0.001) (Mann-Whitney U). They were also associated with increased blood eosinophilia (P = 0.005). There were no significant associations found between ILC2s and serum total IgE and allergic disease. In the CRS with nasal polyps (CRSwNP) population, ILC2s were increased in patients with co-existing asthma (P = 0.03). ILC2s were also correlated with worsening nasal symptom score in CRS (P = 0.04)."
#            ]
# question_buf = ["Are group 2 innate lymphoid cells ( ILC2s ) increased in chronic rhinosinusitis with nasal polyps or eosinophilia?"]

context = " ".join(context_buf)
question = " ".join(question_buf)

prompt = [
  {"role": "user", "content": context}
] + [
  {"role": "user", "content": question}
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
generate_ids1 = model.generate(inputs.input_ids, max_new_tokens=128, do_sample=False)
outputs1 = tokenizer.batch_decode(generate_ids1[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
# print(f"Original output: {outputs1}")
# print("============================")
del model

select_dim = 10000
grad_order = 6
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
    "question": question,
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

with open(f"./case_study/coqa/coqa_protection_W_{select_dim}_N_{grad_order}", "w") as f:
    json.dump(model_tylor_log, f, indent=4)

# ipdb.set_trace()
