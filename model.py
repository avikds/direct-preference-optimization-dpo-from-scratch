"""
Direct Preference Optimization (DPO) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - log_softmax
import numpy as np

def log_softmax(logits, axis=-1):
    # Numerically stable log-softmax:
    # log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    logits = np.asarray(logits)

    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted_logits = logits - max_logits

    log_sum_exp = np.log(
        np.sum(np.exp(shifted_logits), axis=axis, keepdims=True)
    )

    return shifted_logits - log_sum_exp

# Step 2 - softmax
def softmax(logits, axis=-1):
    # Numerically stable softmax:
    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    logits = np.asarray(logits)

    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted_logits = logits - max_logits

    exp_shifted = np.exp(shifted_logits)
    return exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

# Step 3 - gather_token_logprobs
def gather_token_logprobs(log_probs, token_ids):
    # log_probs shape: (B, T, V)
    # token_ids shape: (B, T)
    # Select the vocabulary log-probability corresponding to each token.
    log_probs = np.asarray(log_probs)
    token_ids = np.asarray(token_ids)

    batch_indices = np.arange(log_probs.shape[0])[:, None]
    time_indices = np.arange(log_probs.shape[1])[None, :]

    return log_probs[batch_indices, time_indices, token_ids]

# Step 4 - masked_sequence_logprob
def masked_sequence_logprob(token_logprobs, mask):
    token_logprobs = np.asarray(token_logprobs)
    mask = np.asarray(mask)

    # Zero out masked positions, then sum over the sequence dimension.
    return np.sum(token_logprobs * mask, axis=1)

# Step 5 - init_policy_params
def init_policy_params(vocab_size, d_model, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    scale = 0.02

    return {
        "embed": rng.normal(0.0, scale, size=(vocab_size, d_model)),
        "W_out": rng.normal(0.0, scale, size=(d_model, vocab_size)),
        "b_out": np.zeros(vocab_size),
    }

# Step 6 - policy_token_logits
def policy_token_logits(params, token_ids):
    embed = params["embed"]
    W_out = params["W_out"]
    b_out = params["b_out"]

    hidden = embed[token_ids]
    return hidden @ W_out + b_out

# Step 7 - policy_sequence_logprob
def policy_sequence_logprob(params, token_ids, mask):
    logits = policy_token_logits(params, token_ids)
    log_probs = log_softmax(logits, axis=-1)
    token_logprobs = gather_token_logprobs(log_probs, token_ids)
    return masked_sequence_logprob(token_logprobs, mask)

# Step 8 - sequence_logprob_grad
def sequence_logprob_grad(params, token_ids, mask):
    embed = params["embed"]
    W_out = params["W_out"]
    b_out = params["b_out"]

    logits = policy_token_logits(params, token_ids)
    probs = softmax(logits, axis=-1)

    batch_size, seq_len = token_ids.shape
    vocab_size = W_out.shape[1]

    # d log p(y) / d logits = one_hot(y) - softmax(logits)
    delta = -probs
    batch_idx = np.arange(batch_size)[:, None]
    time_idx = np.arange(seq_len)[None, :]
    delta[batch_idx, time_idx, token_ids] += 1.0

    delta *= mask[..., None]

    hidden = embed[token_ids]

    # Gradient of W_out
    grad_W_out = np.einsum("btd,btv->dv", hidden, delta)

    # Gradient of b_out
    grad_b_out = np.sum(delta, axis=(0, 1))

    # Gradient of embed
    grad_embed = np.zeros_like(embed)
    grad_hidden = delta @ W_out.T

    np.add.at(grad_embed, token_ids, grad_hidden)

    return {
        "embed": grad_embed,
        "W_out": grad_W_out,
        "b_out": grad_b_out,
    }

# Step 9 - bradley_terry_loss
def bradley_terry_loss(reward_chosen, reward_rejected):
    margin = np.asarray(reward_chosen) - np.asarray(reward_rejected)
    return np.mean(np.logaddexp(0.0, -margin))

# Step 10 - reward_accuracy
def reward_accuracy(reward_chosen, reward_rejected):
    reward_chosen = np.asarray(reward_chosen)
    reward_rejected = np.asarray(reward_rejected)

    return np.mean(reward_chosen > reward_rejected)

# Step 11 - build_preference_pairs
def build_preference_pairs(prompts, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    return [
        {
            "prompt": prompts[i],
            "chosen_ids": chosen_ids[i],
            "rejected_ids": rejected_ids[i],
            "chosen_mask": chosen_mask[i],
            "rejected_mask": rejected_mask[i],
        }
        for i in range(len(prompts))
    ]

# Step 12 - sample_preference_batch (not yet solved)
# TODO: implement

# Step 13 - freeze_reference_logprobs (not yet solved)
# TODO: implement

# Step 14 - policy_reference_logratio (not yet solved)
# TODO: implement

# Step 15 - dpo_pair_margin (not yet solved)
# TODO: implement

# Step 16 - dpo_loss (not yet solved)
# TODO: implement

# Step 17 - dpo_loss_grad (not yet solved)
# TODO: implement

# Step 18 - dpo_train_step (not yet solved)
# TODO: implement

# Step 19 - train_dpo (not yet solved)
# TODO: implement

# Step 20 - length_normalized_logprob (not yet solved)
# TODO: implement

# Step 21 - ipo_loss (not yet solved)
# TODO: implement

# Step 22 - implicit_reward (not yet solved)
# TODO: implement

# Step 23 - preference_accuracy (not yet solved)
# TODO: implement

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

