from collections import defaultdict
import random

def get_few_shot_examples(train_dataset, n):
    """
    Select approximately balanced few-shot examples from the training dataset.

    Args:
        train_dataset: The training dataset.
        n: Total number of few-shot examples.

    Returns:
        List of few-shot examples.
    """
    # Group examples by label
    label_to_indices = defaultdict(list)
    for idx, example in enumerate(train_dataset):
        label_to_indices[example["label"]].append(idx)

    # Determine roughly how many examples per label
    n_labels = len(label_to_indices)
    n_per_label = n // n_labels
    few_shot_indices = []

    # Sample approximately balanced examples
    for label, indices in label_to_indices.items():
        sampled_count = min(n_per_label, len(indices))  # Limit to available examples
        few_shot_indices.extend(random.sample(indices, sampled_count))

    # Add additional samples if necessary to meet the total `n`
    remaining_needed = n - len(few_shot_indices)
    if remaining_needed > 0:
        all_indices = [idx for indices in label_to_indices.values() for idx in indices]
        additional_samples = random.sample(all_indices, remaining_needed)
        few_shot_indices.extend(additional_samples)

    # Fetch the few-shot examples
    few_shot_examples = [train_dataset[i] for i in few_shot_indices]
    return few_shot_examples


def format_prompt(few_shot_examples, tokenizer):
    """
    Create a prompt by combining few-shot examples into a single sequence.

    Args:
        few_shot_examples: List of few-shot examples.
        tokenizer: The tokenizer used for decoding the input text.

    Returns:
        str: A formatted prompt combining all few-shot examples.
    """
    prompt_text = ""
    for example in few_shot_examples:
        statement = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        label = example["label"].item()
        prompt_text += f"Statement: {statement} | Label: {label}\n"
    return prompt_text
