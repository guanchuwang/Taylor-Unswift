This is the official codes for Secured Weight Access for Large Language Models via Taylor Expansion.


## Dependency

```angular2html
numpy==1.26.4
torch==2.2.0
datasets==2.17.1
accelerate==0.29.2
scikit-learn==1.4.2
peft==0.10.0
transformers==4.38.1
flash-attn==2.5.7
```


## Test TaylorMLP on the COQA and wikitext-2 datasets:

Use TaylorMLP to secure Llama-3-8B, then generate tokens using input context from the COQA and wikitext-2 datasets:

```bash 
python protection/llama_protection.py 
python protection/llama_wikitext_test.py
```

TaylorMLP takes around 4x latency compared with the original Llama-3-8B.

