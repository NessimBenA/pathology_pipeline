import torch
from flash.core import trainer
from flash.image import ImageClassificationData, ImageClassifier
from torchvision import transforms as T

from typing import Tuple, Union
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.io.input_transform import InputTransform
from dataclasses import dataclass

import config as cfg
import os



# 1. Create the DataModule
torch.manual_seed(12)
@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (224, 224)
    mean: Union[float, Tuple[float, float, float]] = (0.5, 0.5, 0.5)
    std: Union[float, Tuple[float, float, float]] = (0.5, 0.5, 0.5)
    def per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose([T.ToTensor(), T.Resize(self.image_size), T.Normalize(self.mean, self.std)]),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self):
        return T.Compose(
            [
                ApplyToKeys(
                    "input",
                    T.Compose(
                        [
                            T.ToTensor(),
                            T.Normalize(self.mean, self.std),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.RandomAutocontrast(),
                            T.RandomPerspective(),
                            T.RandomRotation(degrees=30)
                        ]
                    ),
                ),
                ApplyToKeys("target", torch.as_tensor),
            ]
        )

datamodule = ImageClassificationData.from_folders(
    train_folder=cfg.TRAINING_SET_PATH,
    val_folder=cfg.TESTING_SET_PATH,
    transform=ImageClassificationInputTransform,
    transform_kwargs=dict(image_size=(224,224)),
    batch_size=16,
    num_workers=40,
    )
from pytorch_lightning.callbacks import ModelCheckpoint


# saves top-K checkpoints based on "val_loss" metric
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_accuracy",
    mode="min",
    dirpath=cfg.PROJECT_SPACE,
    filename="checkpoint{epoch:02d}-{val_loss:.2f}",
)

if __name__ == '__main__':
# 2. Build the task
    model = ImageClassifier(backbone='swin_tiny_patch4_window7_224',labels=datamodule.labels,pretrained=True,learning_rate=0.00032432)
    # 3. Create the trainer and finetune the model
    trainer = trainer.Trainer(max_epochs=4000, accelerator='gpu',devices=-1,strategy='ddp',callbacks=[checkpoint_callback])
    trainer.finetune(model, datamodule=datamodule,strategy=("freeze_unfreeze", 25))

    # 5. Save the model!
    checkpoint_callback.best_model_path
    trainer.save_checkpoint(os.path.join(cfg.PROJECT_SPACE, "weights.pt"))