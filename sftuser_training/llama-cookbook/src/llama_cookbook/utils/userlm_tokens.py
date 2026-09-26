"""Initialize the learned user model's new terminal token (Appendix C.3)."""
import torch


def initialize_terminal_embedding(model, tokenizer):
    token = '<|endconversation|>'
    if token not in tokenizer.get_vocab():
        raise ValueError('Prepare the user tokenizer before training')
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    # Qwen checkpoints can have unused embedding rows beyond the tokenizer's
    # vocabulary, so the new token is not necessarily the final embedding row.
    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        for layer in (model.get_input_embeddings(), model.get_output_embeddings()):
            layer.weight[token_id].copy_(layer.weight[tokenizer.eos_token_id])
