import pytorch_lightning as pl
import torch
import os
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
import torchvision.transforms as T
import argparse
from model import MyLightningModule
import math
from print_model import model_information

def main():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model_name", default= "hf_hub:timm/tf_efficientnet_b7.ns_jft_in1k", type=str, help="Model name")
    parser.add_argument("--num_classes", default=8, type=int, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for StratifiedKFold (default: 5)")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size (default: 8)")
    parser.add_argument("--lr", type=float, default= 5e-4 , help="input learning_rate of AdamW")
    parser.add_argument("--image_path", type=str, default='train\\train', help="input image path example train/train")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--random", type=int, default=42, help="Random stage (default: 42)")

    args = parser.parse_args()

    model_information(args.model_name , args.num_classes, args.img_size,args.train_batch_size)

    #Image path 
    base = os.getcwd()
    image_path = os.path.join(base,args.image_path)

    #Transforms
    transforms = {
    "train": T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ]),
    "test": T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ])
    }

    #Makig datasets
    dataset = datasets.ImageFolder(root=image_path, # target folder of images
                                    transform=transforms["train"], # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    # Get the data and labels
    data = [item[0] for item in dataset.samples]
    labels = [item[1] for item in dataset.samples]

    #defind Tmax for scheduler 
    tmax = math.ceil((len(data)/args.num_folds)*(args.num_folds-1))


    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.random)

    ## Train loop 

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):

        # Split data into train and validation sets
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders for training and validation
        train_dataloader = torch.utils.data.DataLoader(train_set, 
                                                        batch_size=args.train_batch_size, 
                                                        shuffle=True,)
        val_dataloader = torch.utils.data.DataLoader(val_set, 
                                                        batch_size=args.eval_batch_size, 
                                                        shuffle=False,)

        print(f"Fold {fold+1} of {args.num_folds}")

        # Create a PyTorch Lightning module
        lightning_module = MyLightningModule(args.model_name, args.num_classes,T_max=tmax , lr=args.lr)

        # Create PyTorch Lightning data loaders
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False)

        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(max_epochs=args.num_epochs)

        # Train the model
        trainer.fit(lightning_module, train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()