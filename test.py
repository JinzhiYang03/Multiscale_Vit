import torch
import torch.nn as nn
import argparse
import ml_collections
from model import VisionTransformer, VisionTransformerSPP

parser = argparse.ArgumentParser()

parser.add_argument("--name", required=False, default="experiment",
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", required=False, default="Vit",choices=["Vit", "Vit_SPP"],
                    help="Model used")
parser.add_argument("--output_dir", default="output", type=str,
                    help="The output directory where checkpoints will be written.")
parser.add_argument("--img_size", default=28, type=int,
                    help="Resolution size")


parser.add_argument("--dataset", default="mnist",
                    help="Which downstream task.")
parser.add_argument("--train_batch_size", default=64, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=32, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", default=1000, type=int,
                    help="Run prediction on validation set every so many steps."
                          "Will always run one evaluation at the end of training.")
parser.add_argument("--patience", default=5, type=int,
                    help="Patience for early stopping")
parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=100000, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")

args, unknown = parser.parse_known_args()

def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (7, 7)})
    config.hidden_size = 256
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    return config

num_classes = 10
config = get_b16_config()
model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

# Load checkpoint
checkpoint_path = "experiment_checkpoint.pth"
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

