import re
import sys

import torch
from styletokenizer.sadiri.deberta import DebertaV2ForSequence
from transformers import AutoModel, AutoModelForMaskedLM


# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(args, tokenizer=None, evaluate=False):
    if re.search(r"deberta", args.tokenizer):
        model = DebertaV2ForSequence.from_pretrained(args.pretrained_model)
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing:
                model.deberta.encoder.gradient_checkpointing = True

    elif args.sparse:
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
        # use gradient checkpointing to save memory (this can slow training by ~20%)
        print('Using Learned Sparse Retrieval')
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing:
                model.roberta.encoder.gradient_checkpointing = True

    else:
        model = AutoModel.from_pretrained(args.pretrained_model)
        # use gradient checkpointing to save memory (this can slow training by ~20%)
        if hasattr(args, 'gradient_checkpointing'):
            if args.gradient_checkpointing:
                #model.encoder.gradient_checkpointing = True
                model.gradient_checkpointing_enable()


    return model

