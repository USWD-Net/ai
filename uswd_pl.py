from config import *
from models import get_model
from utils.loss import get_loss
from data.dataset import get_dataloader
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import pytorch_lightning as pl

class USWDLightning(pl.LightningModule) :
    def __init__(self) :
        super().__init__()
        self.model = get_model()
        self.loss = get_loss()
        
        self.train_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro')
        self.val_accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES, average='macro')
        
        self.train_f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro')

        self.train_precision = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro')
        self.val_precision = MulticlassPrecision(num_classes=NUM_CLASSES, average='macro')

        self.train_recall = MulticlassRecall(num_classes=NUM_CLASSES, average='macro')
        self.val_recall = MulticlassRecall(num_classes=NUM_CLASSES, average='macro')

        self.train_auroc = MulticlassAUROC(num_classes=NUM_CLASSES, average='macro',
                                            thresholds=None)
        self.val_auroc = MulticlassAUROC(num_classes=NUM_CLASSES, average='macro', thresholds=None)

        
    def forward(self, s1, s2, s3, s4) :
        return self.model(s1, s2, s3, s4)
    
    def training_step(self, batch, batch_idx) :
        s1, s2, s3, s4, labels = batch
        outputs = self.forward(s1, s2, s3, s4)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Accuracy
        self.train_accuracy.update(outputs, labels)
        self.log('train_acc_step', self.train_accuracy, on_step=True, on_epoch=False, logger=True) 
        
        # F1 Score
        self.train_f1.update(outputs, labels)
        self.log('train_f1_step', self.train_f1, on_step=True, on_epoch=False, logger=True)

        # Precision
        self.train_precision.update(outputs, labels)
        self.log('train_precision_step', self.train_precision, on_step=True, on_epoch=False, logger=True)

        # Recall
        self.train_recall.update(outputs, labels)
        self.log('train_recall_step', self.train_recall, on_step=True, on_epoch=False, logger=True)
        
        # AUROC
        probs = torch.softmax(outputs, dim=-1)
        self.train_auroc.update(probs, labels)
        self.log('train_auroc_step', self.train_auroc, on_step=True, on_epoch=False, logger=True)
        
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc_epoch', self.train_accuracy.compute(), on_step=False, on_epoch=True, logger=True)
        self.log('train_f1_epoch', self.train_f1.compute(), on_step=False, on_epoch=True, logger=True)
        self.log('train_precision_epoch', self.train_precision.compute(), on_step=False, on_epoch=True, logger=True)
        self.log('train_recall_epoch', self.train_recall.compute(), on_step=False, on_epoch=True, logger=True)
        self.log('train_auroc_epoch', self.train_auroc.compute(), on_step=False, on_epoch=True, logger=True)
        
        self.train_accuracy.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_auroc.reset()
    
    def validation_step(self, batch, batch_idx) :
        s1, s2, s3, s4, labels = batch
        outputs = self.forward(s1, s2, s3, s4)
        loss = self.loss(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.val_accuracy.update(outputs, labels)
        self.val_f1.update(outputs, labels)
        self.val_precision.update(outputs, labels)
        self.val_recall.update(outputs, labels)
        
        probs = torch.softmax(outputs, dim=-1)
        self.val_auroc.update(probs, labels)
        
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss    
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        return optimizer

    def train_dataloader(self) :
        return get_dataloader(DATA_ROOT, is_val = False)
    
    def val_dataloader(self) :
        return get_dataloader(DATA_ROOT, is_val=True)