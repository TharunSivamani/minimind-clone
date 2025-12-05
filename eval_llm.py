import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')


def init_model(args):

    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))

        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')

    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        print(f'MiniMind Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="MiniMind Model Inference and Chat Interface")

    parser.add_argument(
        '--load_from', 
        default='model', 
        type=str, 
        help="Model loading source. Use 'model' for native PyTorch weights, or specify a directory for Transformers format."
    )

    parser.add_argument(
        '--save_dir', 
        default='out', 
        type=str, 
        help="Directory where model weights are stored."
    )

    parser.add_argument(
        '--weight', 
        default='full_sft', 
        type=str, 
        help="Weight prefix name (options: pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo)."
    )

    parser.add_argument(
        '--lora_weight', 
        default='None', 
        type=str, 
        help="LoRA weight name. Use 'None' to disable LoRA (options: None, lora_identity, lora_medical)."
    )

    parser.add_argument(
        '--hidden_size', 
        default=512, 
        type=int, 
        help="Hidden dimension size (512=Small-26M, 640=MoE-145M, 768=Base-104M)."
    )

    parser.add_argument(
        '--num_hidden_layers', 
        default=8, 
        type=int, 
        help="Number of transformer layers (Small/MoE=8, Base=16)."
    )

    parser.add_argument(
        '--use_moe', 
        default=0, 
        type=int, 
        choices=[0, 1], 
        help="Enable Mixture-of-Experts architecture (0=No, 1=Yes)."
    )

    parser.add_argument(
        '--inference_rope_scaling', 
        default=False, 
        action='store_true', 
        help="Enable RoPE position encoding extrapolation (4× context extension; only solves position limit issues)."
    )

    parser.add_argument(
        '--max_new_tokens', 
        default=8192, 
        type=int, 
        help="Maximum number of tokens to generate (not the true context-length capability of the model)."
    )

    parser.add_argument(
        '--temperature', 
        default=0.85, 
        type=float, 
        help="Sampling temperature (0–1). Higher values produce more randomness."
    )

    parser.add_argument(
        '--top_p', 
        default=0.85, 
        type=float, 
        help="Top-p (nucleus) sampling threshold (0–1)."
    )

    parser.add_argument(
        '--historys', 
        default=0, 
        type=int, 
        help="Number of dialogue history turns to keep (must be even; 0 = no history)."
    )

    parser.add_argument(
        '--device', 
        default='cuda' if torch.cuda.is_available() else 'cpu', 
        type=str, 
        help="Inference device (CPU or CUDA GPU)."
    )

    args = parser.parse_args()

    prompts = [
        "What special skills do you have?",
        "Why is the sky blue?",
        "Write a Python function to compute the Fibonacci sequence.",
        "Explain the basic process of photosynthesis.",
        "If it rains tomorrow, what should I do when going out?",
        "Compare the advantages and disadvantages of cats and dogs as pets.",
        "Explain what machine learning is.",
        "Recommend some traditional Chinese foods."
    ]

    conversation = []
    model, tokenizer = init_model(args)
    
    mode = int(input("[0] Auto Test\n[1] Manual Input\n"))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt_iter = prompts if mode == 0 else iter(lambda: input("User: "), "")

    for prompt in prompt_iter:

        setup_seed(2025)

        if mode == 0:
            print(f"User: {prompt}")
        
        # Keep last N turns if history is enabled
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # Build input template
        template_kwargs = {
            "conversation": conversation,
            "tokenize": False,
            "add_generation_prompt": True
        }

        # Enable thinking tokens only for "reason" models
        if args.weight == 'reason':
            template_kwargs["enable_thinking"] = True

        if args.weight != "pretrain":
            final_input = tokenizer.apply_chat_template(**template_kwargs)
        else:
            final_input = tokenizer.bos_token + prompt

        # Tokenize
        inputs = tokenizer(final_input, return_tensors='pt', truncation=True).to(args.device)

        print("Assistant: ", end=" ")

        # Generate Output
        generated = model.generate(
            inputs = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_new_tokens = args.max_new_tokens,
            do_sample = True,
            streamer = streamer,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            top_p = args.top_p
            temperature = args.temperature,
            frequency_penalty = 1.0
        )

        # Decode assistant reply only (exclude prompt)
        reply = tokenizer.decode(
            generated[0][len(inputs["input_ids"][0]):],
            skip_special_tokens = True
        )

        conversation.append({"role": "assistant", "content": reply})
        print("\n")


if __name__ == "__main__":
    main()