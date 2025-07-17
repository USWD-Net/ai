from utils.seed import set_seed
set_seed()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from uswd_pl import USWDLightning
from config import *

def train() :
    model = USWDLightning()
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        dirpath=LOG_PATH
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu", 
        devices=DEVICES,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=pl.loggers.TensorBoardLogger(LOG_PATH, name=f"{MODEL_NAME}_s{SEED}_b{BATCH_SIZE}_t{ENCODER_TYPE}"),
    )
    trainer.fit(model)
    
if __name__ == '__main__' :
    train()