import argparse
import torch
from torchvision import transforms, datasets
from model import VisionTransformer, VisionTransformerSPP
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ml_collections
import matplotlib.pyplot as plt
from train import get_loader_helper

# parser = argparse.ArgumentParser()

# parser.add_argument("--name", required=False, default="experiment",
#                     help="Name of this run. Used for monitoring.")
# parser.add_argument("--model_type", required=False, default="Vit",choices=["Vit", "Vit_SPP"],
#                     help="Model used")
# parser.add_argument("--output_dir", default="output", type=str,
#                     help="The output directory where checkpoints will be written.")
# parser.add_argument("--img_size", default=28, type=int,
#                     help="Resolution size")


# parser.add_argument("--dataset", default="mnist",
#                     help="Which downstream task.")
# parser.add_argument("--train_batch_size", default=64, type=int,
#                     help="Total batch size for training.")
# parser.add_argument("--eval_batch_size", default=32, type=int,
#                     help="Total batch size for eval.")
# parser.add_argument("--eval_every", default=1000, type=int,
#                     help="Run prediction on validation set every so many steps."
#                           "Will always run one evaluation at the end of training.")
# parser.add_argument("--patience", default=5, type=int,
#                     help="Patience for early stopping")
# parser.add_argument("--learning_rate", default=3e-2, type=float,
#                     help="The initial learning rate for SGD.")
# parser.add_argument("--weight_decay", default=0, type=float,
#                     help="Weight deay if we apply some.")
# parser.add_argument("--num_steps", default=100000, type=int,
#                     help="Total number of training epochs to perform.")
# parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
#                     help="How to decay the learning rate.")
# parser.add_argument("--warmup_steps", default=500, type=int,
#                     help="Step of training to perform learning rate warmup for.")
# parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                     help="Max gradient norm.")

# parser.add_argument("--local_rank", type=int, default=-1,
#                     help="local_rank for distributed training on gpus")
# parser.add_argument('--seed', type=int, default=42,
#                     help="random seed for initialization")
# parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                     help="Number of updates steps to accumulate before performing a backward/update pass.")

# args, unknown = parser.parse_known_args()

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

# num_classes = 10
# config = get_b16_config()
# model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

# # Load checkpoint

# checkpoint_path = "output/experiment_checkpoint.pth"
# model.load_state_dict(torch.load(checkpoint_path))
# model.eval()

def run_inference(args, scaling_factors):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Load model
    config = get_b16_config()
    num_classes = 10

    if args.model_type == "Vit":
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    else:
        model = VisionTransformerSPP(config, args.img_size, zero_head=True, num_classes=num_classes)

    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    accuracies = []

    for scale in scaling_factors:
        args.img_size = 28 * scale
        print(f"Running inference with scaled image size: {args.img_size}")
        
        _, test_loader = get_loader_helper(args, 1)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(iter(test_loader), desc=f"Inference for scale {scale}x"):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                logits, _ = model(images)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        accuracies.append(accuracy)
        print(f"Accuracy for scale {scale}x: {accuracy * 100:.2f}%")

    return scaling_factors, accuracies

def plot_accuracy(args, scaling_factors, accuracies, with_spp=False):
    save_path=args.save_path
    plt.figure(figsize=(10, 6))
    plt.plot(scaling_factors, accuracies, marker='o')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Accuracy')
    
    if with_spp:
        plt.title('Accuracy vs. Scaling Factor for MNIST (with SPP)')
        save_path="accuracy_vs_scaling_spp.png"
    else:
        plt.title('Accuracy vs. Scaling Factor for MNIST (no SPP)')
        
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="Vit", choices=["Vit", "Vit_SPP"])
    parser.add_argument("--img_size", default=28, type=int)
    parser.add_argument("--eval_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint_path", default="output/experiment_checkpoint.pth")
    parser.add_argument("--save_path", default="accuracy_vs_scaling.png")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    scaling_factors = [1, 2, 4, 8] 
    scaling_factors, accuracies = run_inference(args, scaling_factors)
    plot_accuracy(args, scaling_factors, accuracies, (args.model_type == "Vit_SPP"))


if __name__ == "__main__":
    main()
