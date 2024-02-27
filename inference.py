from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from transformers import LlamaTokenizerFast

from mistral.model import Transformer ,ModelArgs






def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)

@torch.inference_mode()
def generate(prompt: str, model: Transformer, tokenizer: LlamaTokenizerFast,
              *, max_tokens: int,  temperature: float):
    

    model= model.eval()
    input_ids, seq_lengths= [], []

    tokenize_prompt = tokenizer.encode(prompt)

    input_ids.extend(tokenize_prompt)
    seq_lengths.append(len(tokenize_prompt))

    token_tensor = torch.tensor(input_ids, dtype=torch.long).to(device)

    for _ in range(max_tokens):
        prelogits = model.forward(input_ids=token_tensor, seqlens=seq_lengths)
        logits = torch.log_softmax(prelogits, dim=-1)
        next_token = sample(logits,temperature=temperature, top_p=0.8)
        predicted_text = tokenizer.decode(next_token)
        print(f' {predicted_text} ')
        new_text = f'{prompt} {predicted_text}'
        tokenize_text = tokenizer.encode(new_text)
        input_ids.clear
        seq_lengths.clear
        input_ids.extend(tokenize_text)
        seq_lengths.append(len(tokenize_text))
        token_tensor = torch.tensor(input_ids, dtype=torch.long).to(device)



if __name__ == "__main__":
   path = "../../mar-tokenizer"
   tokenizer_loaded = LlamaTokenizerFast.from_pretrained(path)
   tokenizer_loaded.add_special_tokens({'pad_token': '[PAD]'})
   device = "cuda"

   arg = ModelArgs(dim=1024,
                n_layers=12,
                head_dim=6,
                hidden_dim=4096,
                n_heads=8,
                n_kv_heads=4,
                norm_eps=1e-6,
                max_batch_size=8,
                vocab_size=200,
                sliding_window=4,
                )

   model = Transformer(args=arg).to(device)

   model.load_state_dict(torch.load('weights/my-lm_epochs-1000_v2.pth'))

   prompt = input(str("Enter some text to generate"))

   generate(prompt=prompt,
            model=model,
            tokenizer=tokenizer_loaded,
            max_tokens=20,
            temperature=0.9,
            )


