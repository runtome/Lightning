import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model_name", default= "hf_hub:timm/tf_efficientnet_b7.ns_jft_in1k", type=str, help="Model name")
    parser.add_argument("--num_classes", type=int,  default=8 ,help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for StratifiedKFold (default: 5)")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size (default: 8)")
    parser.add_argument("--lr", type=float, default= 5e-4 , help="input learning_rate of AdamW")
    parser.add_argument("--image_path", type=str, default='train\\train', help="input image path example train/train")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--random", type=int, default=42, help="Random stage (default: 42)")

    args = parser.parse_args()

    return args