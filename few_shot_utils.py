from collections import defaultdict
import random
import torch
import torch.nn.functional as F

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


def evaluate_model_with_few_shot_prompt(
    model,
    tokenizer,
    few_shot_prompt,
    test_loader,
    max_examples=100,
    max_length=2048
):
    """
    Evaluates the model on a test dataset using a few-shot prompt.

    Args:
        model: Pretrained causal language model.
        tokenizer: Tokenizer associated with the model.
        few_shot_prompt (str): Few-shot prompt to prepend to each test statement.
        test_loader: DataLoader for the test dataset.
        max_examples (int): Maximum number of test examples to process.
        max_length (int): Maximum input length for tokenization.

    Returns:
        test_predictions (list): Predicted labels.
        test_labels (list): Ground truth labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label_token_ids = [tokenizer.convert_tokens_to_ids(str(i)) for i in range(6)]
    assert all(token_id is not None for token_id in label_token_ids), "Some labels are not in the vocabulary!"
    print("Label Token IDs:", label_token_ids)

    tokenized_input = tokenizer(few_shot_prompt, truncation=False, return_tensors="pt")
    print("Prompt tokenized length:", len(tokenized_input["input_ids"][0]))

    test_predictions, test_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            for i in range(len(batch["input_ids"])):
                if len(test_predictions) >= max_examples:
                    break

                statement = batch["statement"][i]
                subject = batch.get("subject", ["Unknown"])[i]
                speaker = batch.get("speaker", ["Unknown"])[i]
                job_title = batch.get("job_title", ["Unknown"])[i]
                state_info = batch.get("state_info", ["Unknown"])[i]
                party_affiliation = batch.get("party_affiliation", ["Unknown"])[i]
                context = batch.get("context", ["Unknown context"])[i]

                test_input_text = (
                    few_shot_prompt +
                    f"Context: This statement was made by {speaker}, a {job_title} from {state_info} "
                    f"and a member of the {party_affiliation} party, in {context}. Subject: {subject}.\n"
                    f"Statement: \"{statement}\" | Label:"
                )

                test_input = tokenizer(
                    test_input_text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )

                input_ids = test_input["input_ids"].to(device)

                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]

                label_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                label_log_likelihoods = torch.tensor(
                    [F.log_softmax(label_logits, dim=-1)[0, token_id].item() for token_id in label_token_ids],
                    device=device
                )

                prediction = torch.argmax(label_log_likelihoods).item()
                test_predictions.append(prediction)
                test_labels.append(batch["label"][i].item())

    return test_predictions, test_labels


def format_prompt_with_context(few_shot_examples):
    """
    Generate a few-shot prompt using enriched statements with simplified context.
    """
    prompt = ""
    for example in few_shot_examples:
        statement = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
        label = example["label"]
        subject = example.get("subject", "Unknown")
        speaker = example.get("speaker", "Unknown")
        job_title = example.get("job_title", "Unknown")
        state_info = example.get("state_info", "Unknown")
        party_affiliation = example.get("party_affiliation", "Unknown")
        context = example.get("context", "Unknown context")

        enriched_statement = (
            f"Context: This statement was made by {speaker}, a {job_title} from {state_info} "
            f"and a member of the {party_affiliation} party, in {context}. Subject: {subject}.\n"
            f"Statement: \"{statement}\" | Label: {label}\n\n"
        )

        prompt += enriched_statement
    return prompt
