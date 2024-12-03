import logging
import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.functional import pad
from model import VisionTransformer, VisionTransformerSPP
import ml_collections
import math
import random
from PIL import Image

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

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class RandomScaleTransform:
    def __init__(self, scale_range=(1, 8)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale_factor = random.uniform(*self.scale_range)
        new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
        img = img.resize(new_size, Image.LANCZOS)
        return img

def pad_collate(batch):
    max_height = max(item[0].shape[1] for item in batch)
    max_width = max(item[0].shape[2] for item in batch)

    # pad all images to match max_height and max_width
    padded_images = []
    labels = []
    for img, label in batch:
        padding = (0, 0, max_width - img.shape[2], max_height - img.shape[1])  # (left, top, right, bottom)
        padded_images.append(pad(img, padding))
        labels.append(label)

    return torch.stack(padded_images), torch.tensor(labels)

def get_loader_helper(args, scale=1):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_train_mnist = transforms.Compose([
        transforms.Resize((args.img_size * scale, args.img_size * scale)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test_mnist = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == "mnist":
        trainset = datasets.MNIST(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train_mnist)
        testset = datasets.MNIST(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test_mnist) if args.local_rank in [-1, 0] else None

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              collate_fn=pad_collate,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader

def get_loader(args):
    train_loaders = []
    test_loaders = []
    if args.model_type == 'Vit_SPP':
        s = 5
    else:
        s = 2
    for i in range(1, s):
        train_loader, test_loader = get_loader_helper(args, i)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    return train_loaders, test_loaders
    

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = get_b16_config()

    num_classes = 10
    if args.model_type == "Vit":
        model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
        if args.training_mode == 'finetune':
            model.load_state_dict(torch.load(args.pretrain_dir))
    else:
        model = VisionTransformerSPP(config, args.img_size, zero_head=True, num_classes=num_classes)
        if args.training_mode == 'finetune':
            model.load_state_dict(torch.load(args.pretrain_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loaders, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loaders)*len(test_loaders[0]))
    logger.info("  Num steps = %d", len(test_loaders))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    for test_loader in test_loaders:
        epoch_iterator = tqdm(test_loader,
                              desc="Validating... (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        loss_fct = torch.nn.CrossEntropyLoss()
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            with torch.no_grad():
                logits = model(x)[0]
    
                eval_loss = loss_fct(logits, y)
                eval_losses.update(eval_loss.item())
    
                preds = torch.argmax(logits, dim=-1)
    
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_label.append(y.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(
                    all_preds[0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[0] = np.append(
                    all_label[0], y.detach().cpu().numpy(), axis=0
                )
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
    
    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader_ls, test_loader_ls = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc, patience_counter = 0, 0, 0

    while True:
        model.train()
        if args.model_type == 'Vit_SPP':
            epoch_iterator_ls = []
            for i in range(len(train_loader)):
                epoch_iterator = tqdm(train_loader_ls[i],
                epoch_iterator = tqdm(train_loader[i],
                                    desc="Training (X / X Steps) (loss=X.X)",
                                    bar_format="{l_bar}{r_bar}",
                                    dynamic_ncols=True,
                                    disable=args.local_rank not in [-1, 0])
                epoch_iterator_ls.append(epoch_iterator)
            
            from itertools import zip_longest
            for step, batches in enumerate(zip_longest(*epoch_iterator_ls, fillvalue=None)):
                for i, batch in enumerate(batches):
                    if batch is None:
                        # Skip if the iterator is exhausted
                        continue

                    # Move data to the correct device
                    batch = tuple(t.to(args.device) for t in batch)
                    x, y = batch
                    loss = model(x, y)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        losses.update(loss.item() * args.gradient_accumulation_steps)

                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        epoch_iterator_ls[i].set_description(
                            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                        )
                        if args.local_rank in [-1, 0]:
                            writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                            writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                            accuracy = valid(args, model, writer, test_loader_ls, global_step)
                            if accuracy > best_acc:
                                save_model(args, model)
                                best_acc = accuracy
                                patience_counter = 0  # Reset patience counter if accuracy improves
                            else:
                                patience_counter += 1
                                logger.info(f"No improvement in accuracy for {patience_counter} evaluations.")

                            # Early stopping
                            if patience_counter >= args.patience:
                                logger.info("Early stopping triggered!")
                                return

                            model.train()

                        if global_step % t_total == 0:
                            break
        else:

            epoch_iterator = tqdm(train_loader_ls[0],
                                desc="Training (X / X Steps) (loss=X.X)",
                                bar_format="{l_bar}{r_bar}",
                                dynamic_ncols=True,
                                disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
                loss = model(x, y)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses.update(loss.item() * args.gradient_accumulation_steps)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                    )
                    if args.local_rank in [-1, 0]:
                        writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                        writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                        accuracy = valid(args, model, writer, test_loader_ls, global_step)
                        if accuracy > best_acc:
                            save_model(args, model)
                            best_acc = accuracy
                            patience_counter = 0  # Reset patience counter if accuracy improves
                        else:
                            patience_counter += 1
                            logger.info(f"No improvement in accuracy for {patience_counter} evaluations.")
                        
                        # Early stopping
                        if patience_counter >= args.patience:
                            logger.info("Early stopping triggered!")
                            return

                        model.train()

                    if global_step % t_total == 0:
                        break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False, default="experiment",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", required=False, default="Vit",choices=["Vit", "Vit_SPP"],
                        help="Model used")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=28, type=int,
                        help="Resolution size")
    parser.add_argument("--random_resize", default=False, type=bool,
                        help="Random Resize On")
    parser.add_argument("--training_mode", default="train", choices=["train", "finetune"],
                        help="Training mode")
    parser.add_argument("--pretrain_dir", required=False, type=str,
                        help="The pretrained model directory where checkpoints will be loaded.")
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

    # Setup CUDA, GPUg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)

    
if __name__ == "__main__":
    main()
