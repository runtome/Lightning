import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torch.optim import AdamW
import evaluate

    
class MyLightningModule(pl.LightningModule):
    def __init__(self, model_name , num_classes, T_max):
        super(MyLightningModule, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.T_max = T_max
        self.learning_rate = 5e-4
        self.criterion = nn.CrossEntropyLoss()
        self.metric = evaluate.load("f1")
        self.save_hyperparameters(ignore=['model'])
        self.metric = evaluate.load("f1") ##Add evaluate 
        self.model = timm.create_model(model_name=self.model_name, pretrained=True,num_classes=self.num_classes)

    def forward(self, x):
        # Define the forward pass of your model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        ##Add Evaluate log 
        pred = outputs.argmax(-1)
        acc = self.metric.compute(predictions=pred, references=targets, average='macro')['f1']
        self.log('train_accuracy', acc , prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)  # Log the loss for visualization
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        ##Add Evaluate log 
        pred = outputs.argmax(-1)
        acc = self.metric.compute(predictions=pred, references=targets, average='macro')['f1']
        self.log('val_accuracy', acc , prog_bar=True)

        self.log('val_loss', loss,prog_bar=True)  # Log the validation loss
        return loss
    
    

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)
        return [optimizer], [scheduler]


