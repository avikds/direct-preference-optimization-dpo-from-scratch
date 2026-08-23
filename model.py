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

# Step 12 - sample_preference_batch
def sample_preference_batch(pairs, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if not pairs or batch_size <= 0:
        raise ValueError("pairs must be non-empty and batch_size must be positive")

    replace = batch_size > len(pairs)
    indices = rng.choice(len(pairs), size=batch_size, replace=replace)
    sampled = [pairs[i] for i in indices]

    batch = {
        "chosen_ids": np.stack([pair["chosen_ids"] for pair in sampled]),
        "rejected_ids": np.stack([pair["rejected_ids"] for pair in sampled]),
        "chosen_mask": np.stack([pair["chosen_mask"] for pair in sampled]),
        "rejected_mask": np.stack([pair["rejected_mask"] for pair in sampled]),
    }

    if "prompt" in sampled[0]:
        batch["prompt"] = np.array([pair["prompt"] for pair in sampled])

    return batch

# Step 13 - freeze_reference_logprobs
def freeze_reference_logprobs(ref_params, pairs):
    frozen = []

    for pair in pairs:
        chosen_logprob = policy_sequence_logprob(ref_params, pair["chosen_ids"][None, :], pair["chosen_mask"][None, :])[0]
        rejected_logprob = policy_sequence_logprob(ref_params, pair["rejected_ids"][None, :], pair["rejected_mask"][None, :])[0]

        frozen.append({
            "chosen": chosen_logprob,
            "rejected": rejected_logprob,
        })

    return frozen

# Step 14 - policy_reference_logratio
def policy_reference_logratio(policy_logprob, reference_logprob):
    return np.asarray(policy_logprob) - np.asarray(reference_logprob)

# Step 15 - dpo_pair_margin
def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    chosen_ratio = np.asarray(policy_logprob_chosen) - np.asarray(ref_logprob_chosen)
    rejected_ratio = np.asarray(policy_logprob_rejected) - np.asarray(ref_logprob_rejected)
    return beta * (chosen_ratio - rejected_ratio)

# Step 16 - dpo_loss
def dpo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    margins = dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta)
    return float(np.mean(np.logaddexp(0.0, -margins)))

# Step 17 - dpo_loss_grad
def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    chosen_logprob = policy_sequence_logprob(
        params, batch["chosen_ids"], batch["chosen_mask"]
    )
    rejected_logprob = policy_sequence_logprob(
        params, batch["rejected_ids"], batch["rejected_mask"]
    )

    ref_chosen = np.asarray(ref_logprobs_batch["chosen"])
    ref_rejected = np.asarray(ref_logprobs_batch["rejected"])

    margins = dpo_pair_margin(
        chosen_logprob,
        rejected_logprob,
        ref_chosen,
        ref_rejected,
        beta,
    )

    loss = float(np.mean(np.logaddexp(0.0, -margins)))

    batch_size = len(margins)

    # d/dm [-log(sigmoid(m))] = -sigmoid(-m)
    sigmoid_neg_margin = np.empty_like(margins, dtype=float)
    positive = margins >= 0
    sigmoid_neg_margin[positive] = np.exp(-margins[positive]) / (1.0 + np.exp(-margins[positive]))
    sigmoid_neg_margin[~positive] = 1.0 / (1.0 + np.exp(margins[~positive]))

    coeff = -beta * sigmoid_neg_margin / batch_size

    grads = {
        key: np.zeros_like(value)
        for key, value in params.items()
    }

    for i in range(batch_size):
        chosen_grads = sequence_logprob_grad(
            params,
            batch["chosen_ids"][i:i + 1],
            batch["chosen_mask"][i:i + 1],
        )

        rejected_grads = sequence_logprob_grad(
            params,
            batch["rejected_ids"][i:i + 1],
            batch["rejected_mask"][i:i + 1],
        )

        for key in params:
            grads[key] += coeff[i] * (
                chosen_grads[key] - rejected_grads[key]
            )

    return loss, grads

# Step 18 - dpo_train_step
def dpo_train_step(params, batch, ref_logprobs_batch, beta, learning_rate):
    loss, grads = dpo_loss_grad(
        params,
        batch,
        ref_logprobs_batch,
        beta,
    )

    updated_params = {
        key: params[key] - learning_rate * grads[key]
        for key in params
    }

    metrics = {
        "loss": float(loss),
    }

    return updated_params, metrics

# Step 19 - train_dpo
def train_dpo(params, pairs, ref_logprobs, beta, learning_rate, num_steps, batch_size, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    current_params = {key: value.copy() for key, value in params.items()}
    history = []

    for step in range(num_steps):
        n = len(pairs)
        indices = rng.choice(n, size=batch_size, replace=(batch_size > n))

        batch = {
            "chosen_ids": np.stack([np.asarray(pairs[i]["chosen_ids"]) for i in indices]),
            "rejected_ids": np.stack([np.asarray(pairs[i]["rejected_ids"]) for i in indices]),
            "chosen_mask": np.stack([np.asarray(pairs[i]["chosen_mask"]) for i in indices]),
            "rejected_mask": np.stack([np.asarray(pairs[i]["rejected_mask"]) for i in indices]),
        }

        ref_batch = {
            "chosen": np.asarray(ref_logprobs["chosen"])[indices],
            "rejected": np.asarray(ref_logprobs["rejected"])[indices],
        }

        current_params, metrics = dpo_train_step(
            current_params,
            batch,
            ref_batch,
            beta,
            learning_rate,
        )

        history.append({
            "loss": float(metrics["loss"]),
            "step": step,
        })

    return current_params, history

# Step 20 - length_normalized_logprob
def length_normalized_logprob(seq_logprob, mask):
    seq_logprob = np.asarray(seq_logprob, dtype=float)
    mask = np.asarray(mask)

    token_counts = np.sum(mask, axis=1)
    return seq_logprob / token_counts

# Step 21 - ipo_loss
def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    policy_ratio = np.asarray(policy_logprob_chosen) - np.asarray(policy_logprob_rejected)
    ref_ratio = np.asarray(ref_logprob_chosen) - np.asarray(ref_logprob_rejected)
    margin = policy_ratio - ref_ratio
    target = 1.0 / (2.0 * beta)

    return float(np.mean((margin - target) ** 2))

# Step 22 - implicit_reward
def implicit_reward(policy_logprob, reference_logprob, beta):
    return beta * (np.asarray(policy_logprob) - np.asarray(reference_logprob))

# Step 23 - preference_accuracy
def preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    chosen_reward = implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    rejected_reward = implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)

    return float(np.mean(chosen_reward > rejected_reward))

# Step 24 - kl_to_reference (not yet solved)
# TODO: implement

# Step 25 - reward_margin_stats (not yet solved)
# TODO: implement

# Step 26 - evaluate_dpo (not yet solved)
# TODO: implement

# Step 27 - run_dpo_pipeline (not yet solved)
# TODO: implement

