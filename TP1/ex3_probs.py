# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import math

# model = GPT2LMHeadModel.from_pretrained("gpt2")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# phrase = "Artificial intelligence is fascinating."
# inputs = tokenizer(phrase, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits  # (1, seq_len, vocab)

# # Convert logits to probabilities using softmax
# probs = torch.softmax(logits, dim=-1)

# # Display P(token_t | tokens_) for t >= 1
# input_ids = inputs["input_ids"][0]
# for t in range(1, len(input_ids)):
#     tok_id = input_ids[t].item()
#     p = probs[0, t-1, tok_id].item()
#     tok_txt = tokenizer.decode([tok_id])
#     print(t, repr(tok_txt), f"{p:.3e}")

# # Computing perplexity
# log_probs = torch.log_softmax(logits, dim=-1)
# input_ids = inputs["input_ids"][0]

# total_logp = 0.0
# n = 0

# for t in range(1, len(input_ids)):
#     tok_id = input_ids[t].item()
#     lp = log_probs[0, t-1, tok_id].item()
#     total_logp += lp
#     n += 1

# avg_neg_logp = -total_logp / n
# ppl = math.exp(avg_neg_logp)

# print("total_logp:", total_logp)
# print("avg_neg_logp:", avg_neg_logp)
# print("perplexity:", ppl)


import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def token_probs(model, tokenizer, sentence: str):
    """
    Retourne pour chaque token t>=1 : P(x_t | x_<t) et logP(x_t | x_<t),
    + total_logp, perplexity.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0]  

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits 

    probs = torch.softmax(logits, dim=-1)             
    log_probs = torch.log_softmax(logits, dim=-1)    

    per_token = []
    total_logp = 0.0
    n = 0

    for t in range(1, len(input_ids)):
        tok_id = input_ids[t].item()
        p = probs[0, t - 1, tok_id].item()
        lp = log_probs[0, t - 1, tok_id].item()
        tok_txt = tokenizer.decode([tok_id])

        per_token.append((t, tok_txt, tok_id, p, lp))
        total_logp += lp
        n += 1

    avg_neg_logp = - total_logp / n
    ppl = math.exp(avg_neg_logp)

    print('-----------------------------------------------')
    print(sentence)
    print("total_logp:", total_logp)
    print("avg_neg_logp:", avg_neg_logp)
    print("perplexity:", ppl)

def main():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.eval()

    phrase = "Artificial intelligence is fascinating."
    token_probs(model, tokenizer, phrase)

    phrase2 = "Artificial fascinating intelligence is."
    token_probs(model, tokenizer, phrase2)

    phrasefr = "L'intelligence artificielle est fascinante."
    token_probs(model, tokenizer, phrasefr)

    prefix = "Artificial intelligence is"
    inp = tokenizer(prefix, return_tensors="pt")

    with torch.no_grad():
        out = model(**inp)
        logits2 = out.logits  

    last_index = logits2.shape[1] - 1
    last_logits = logits2[0, last_index, :]  
    last_probs = torch.softmax(last_logits, dim=-1)

    topk = 10
    vals, idx = torch.topk(last_probs, k=topk)

    print("\n Top-10 \n", repr(prefix))
    for p, tid in zip(vals.tolist(), idx.tolist()):
        print(repr(tokenizer.decode([tid])), f"{p:.3e}")

if __name__ == "__main__":
    main()