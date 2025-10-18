import sys
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch import nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import GemmaForTokenClassification, LlamaForTokenClassification, Qwen2ForTokenClassification

# Set up training arguments
training_args = TrainingArguments(
    output_dir=sys.argv[4],
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10,
    learning_rate=2e-6,
    fp16=True,
    fp16_full_eval=True,
    weight_decay=0.01,
    warmup_ratio=0.05,
    max_steps=0, # note: only for small datasize
    deepspeed="ds_config.json",
    save_only_model=True,
    report_to="none",
    seed=0
)
HEAD_LR = 1e-5 # special learning rate for head

# Load model, tokenizer, and dataset
model_name = sys.argv[1]
config = AutoConfig.from_pretrained(model_name)
# config.classifier_dropout = 0.1
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")

if config.num_labels != 1:
    config.num_labels = 1

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class MyLlamaForTokenClassification(LlamaForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.cnt = 0 # ugly implementation of lr balancing, current steps = self.cnt / accum steps

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = super().forward(input_ids, attention_mask, labels=None, return_dict=True, **kwargs)
        label_mask = torch.logical_and(attention_mask != 0, labels != 0)
        logits = outputs.logits.squeeze()

        alpha = max(min(round(self.cnt / 768 / 4, 2), 1), 0) # 768 steps
        
        loss_fct = MSELoss()
        outputs["loss"] = (1 - alpha) * loss_fct(logits[label_mask], labels[label_mask].to(torch.float16))
        
        returns = []
        for _logits, _labels, _label_mask in zip(logits.detach(), labels, label_mask):
            _values = _logits[_label_mask].unsqueeze(0)
            _rewards = _labels.clone()[_label_mask].unsqueeze(0)
            _rewards[:, :-1] = 0
            _action_mask = torch.ones_like(_label_mask)[_label_mask].unsqueeze(0)
            _returns = self.calc_return(_values, _rewards, _action_mask).squeeze()
            returns.append(_returns)
        returns = torch.cat(returns, dim=0)
        outputs["loss"] += alpha * loss_fct(logits[label_mask], returns.to(torch.float16))
        
        self.cnt += 1
        return outputs

    # GAE
    def calc_return(self, values, rewards, action_mask):
        gamma, lambd = 1, 0.8
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        values = torch.clamp(values, min=-1.0, max=1.0)
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return torch.clamp(returns, min=-1.0, max=1.0)

model = MyLlamaForTokenClassification.from_pretrained(model_name, config=config, torch_dtype=torch.float16)

if config.pad_token_id is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    config.pad_token = "[PAD]"
    config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = "[PAD]"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    print("without pad token, append [PAD], set pad token id", tokenizer.pad_token_id)


train_file = sys.argv[2]
valid_file = sys.argv[3]
dataset = load_dataset(
    "json",
    data_files={"train": train_file,
                "valid": valid_file},
)

ADD_EOS_TOKEN=False
# eos_token = "<|eot_id|>"
eos_token = tokenizer.eos_token
def add_eos_token(text):
    if not text.endswith(eos_token):
        text += eos_token
    global ADD_EOS_TOKEN
    if not ADD_EOS_TOKEN:
        print("ADD EOS:", text)
        ADD_EOS_TOKEN = True
    return text

input_column = "text"
label_column = "label"
def preprocess_function(examples):
    # model depend
    split_token_ids = [16533, 25] # Llama
    inputs, targets = [], []
    for i in range(len(examples[input_column])):
        inputs.append(add_eos_token(examples[input_column][i]))
        targets.append(1 if examples[label_column][i] > 0 else -1)

    model_inputs = tokenizer(inputs, padding="max_length", max_length=1536, truncation=True, add_special_tokens=True)
    new_model_inputs = {k: [] for k in model_inputs.keys()}
    new_model_inputs["labels"] = []
    skip_cnt = 0
    for idx, target in enumerate(targets):
        input_ids = model_inputs["input_ids"][idx]
        if input_ids[-1] != tokenizer.eos_token_id and input_ids[-1] != tokenizer.pad_token_id:
            skip_cnt += 1
            continue
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        # find split_token_ids in input_ids
        for i in range(len(input_ids) - len(split_token_ids)):
            if all([input_ids[i + j] == token_id for j, token_id in enumerate(split_token_ids)]):
                labels = [0] * (i + len(split_token_ids) - 1) + [target] + \
                         [target if token.endswith("ĊĊ") else 0 for token in tokens[i + len(split_token_ids):]] # endswith "\n"
                token_cnt = sum([1 if token != tokenizer.pad_token else 0 for token in tokens])
                labels[token_cnt - 1] = target
                for k, v in new_model_inputs.items():
                    if k == "labels":
                        v.append(labels)
                    else:
                        v.append(model_inputs[k][idx])
                # print(token_cnt, [(token, label) for token, label in zip(tokens, labels)])
                break
    # print("skip_cnt:", skip_cnt)
    return new_model_inputs

with training_args.main_process_first(desc="dataset map tokenizer"):
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=[input_column, label_column],
        desc="Running tokenizer on train dataset",
    )

with training_args.main_process_first(desc="dataset map tokenizer"):
    valid_dataset = dataset["valid"].map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=[input_column, label_column],
        desc="Running tokenizer on valid dataset",
    )

# for param in model.model.parameters():
#     param.requires_grad = False


from transformers.optimization import Adafactor, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled
class MyTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            head_parameters = [name for name, _ in self.model.named_parameters() if "score" in name]
            # print("*** head parameters ***")
            # print(head_parameters)
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n not in head_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in head_parameters],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if n in head_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": HEAD_LR,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return batch

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)

# Initialize Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()