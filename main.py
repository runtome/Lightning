import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
import torchvision.transforms as T
import evaluate
import argparse



class CustomModule(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = evaluate.load("f1")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        # Log training loss and accuracy
        self.log('train_loss', loss)
        # acc = self.metric(outputs, y)
        pred = outputs.argmax(-1)
        acc = self.metric.compute(predictions=pred, references=y, average='macro')['f1']
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        # Log validation loss and accuracy
        self.log('val_loss', loss)
        # acc = self.metric(outputs, y)
        # print(f'Output /n {outputs.argmax(-1)}')
        # print(f'Y /n {y}')
        pred = outputs.argmax(-1)
        acc = self.metric.compute(predictions=pred, references=y, average='macro')['f1']
        self.log('val_accuracy', acc)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model_name", default= "hf_hub:timm/tf_efficientnet_b7.ns_jft_in1k", type=str, help="Model name")
    parser.add_argument("--num_classes", type=int, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--base_path", type=str, help="Base path for data")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for StratifiedKFold (default: 5)")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size (default: 8)")
    parser.add_argument("--image_path", type=str, default='train/train', help="input image path example train/train")

    args = parser.parse_args()

    #Using argus
    model_name = args.model_name
    num_classes = args.num_classes
    num_epochs = args.num_epochs
    base_path = args.base_path
    num_folds = args.num_folds
    train_batch_size =args.train_batch_size
    eval_batch_size = args.eval_batch_size
    path = args.image_path

    #Image path 
    base = os.getcwd()
    image_path = os.path.join(base,path)

    #Transforms
    transforms = {
    "train": T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ]),
    "test": T.Compose([
        T.Resize((224, 224)),
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


    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    


    #Loop training 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):

        # Split data into train and validation sets
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders for training and validation
        train_dataloader = torch.utils.data.DataLoader(train_set, 
                                                        batch_size=train_batch_size, 
                                                        shuffle=True,)
        val_dataloader = torch.utils.data.DataLoader(val_set, 
                                                        batch_size=eval_batch_size, 
                                                        shuffle=False,)

        print(f"Fold {fold+1} of {num_folds}")

        # Create a PyTorch Lightning module
        model = CustomModule(model_name, num_classes)

        # Create PyTorch Lightning data loaders
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=eval_batch_size, shuffle=False)

        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(max_epochs=num_epochs)

        # Train the model
        trainer.fit(model, train_dataloader, val_dataloader)
