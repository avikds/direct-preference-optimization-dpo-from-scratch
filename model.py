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

# Step 6 - policy_token_logits (not yet solved)
# TODO: implement

# Step 7 - policy_sequence_logprob (not yet solved)
# TODO: implement

# Step 8 - sequence_logprob_grad (not yet solved)
# TODO: implement

# Step 9 - bradley_terry_loss (not yet solved)
# TODO: implement

# Step 10 - reward_accuracy (not yet solved)
# TODO: implement

# Step 11 - build_preference_pairs (not yet solved)
# TODO: implement

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

