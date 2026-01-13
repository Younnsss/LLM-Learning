import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model.eval()

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt") 

print("SEED:", SEED)
print("PROMPT:", repr(prompt))
print("-" * 100)

print("\n------------------------------Greedy-------------------------------")
output_greedy = model.generate(
    **inputs,
    max_length=50,
)
txt_greedy = tokenizer.decode(output_greedy[0], skip_special_tokens=True)
print(txt_greedy)
print("-" * 100)

def generate_once(seed, repetition_penalty=None, temperature=0.7):
    torch.manual_seed(seed)
    kwargs = dict(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
    )
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty

    out = model.generate(**kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)

print("\n-------------------------------Sampling----------------------------------------")
# for s in [1, 2, 3, 4, 5]:
#     print("SEED", s)
#     print(generate_once(s))
#     print("-" * 60)


print("----------------------------Repetition penalty comparison------------------------")
# no_pen = generate_once(42, repetition_penalty=None, temperature=0.7)
# pen = generate_once(42, repetition_penalty=2.0, temperature=0.7)
# print("\n Without repetition_penalty")
# print(no_pen)
# print("\n With repetition_penalty=2.0")
# print(pen)


print("\n-------------------------------Temperature----------------------------------------")
# print('temperature : 0.1')
# print(generate_once(42, temperature=0.1))
# print('temperature : 2.0')
# print(generate_once(42, temperature=2.0))


print("\n-----------------------------Beam search-------------------------------------------")
out_beam5 = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
txt_beam5 = tokenizer.decode(out_beam5[0], skip_special_tokens=True)
print(txt_beam5)

print("=" * 25)

out_beam5 = model.generate(
    **inputs,
    max_length=50,
    num_beams=10,
    early_stopping=True
)
txt_beam5 = tokenizer.decode(out_beam5[0], skip_special_tokens=True)
print(txt_beam5)

print("=" * 25)

out_beam5 = model.generate(
    **inputs,
    max_length=50,
    num_beams=20,
    early_stopping=True
)
txt_beam5 = tokenizer.decode(out_beam5[0], skip_special_tokens=True)
print(txt_beam5)
