import os
from skimage import io
import torchvision
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class OrangesDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = data_dir
        self.list_images = list(df["image"])
        self.list_labels = list(df["label"])
        self.label_to_categorical = {
            'FreshOrange': 0,
            'RottenOrange': 1
        }
        self.transform = transform

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.list_labels[idx], self.list_images[idx])
        image = io.imread(img_path)
        
        label = self.label_to_categorical[self.list_labels[idx]]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class OrangesDataModule(LightningDataModule):
    def __init__(self, data_dir, df, batch_size):

        self.data_dir = data_dir
        
        self.df_train = df[df['stage']=='train']
        self.df_validation = df[df['stage']=='validation']
        self.df_test = df[df['stage']=='test']
        
        self.transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(45),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.batch_size = batch_size
        self.num_workers = 16
        self.prepare_data_per_node = True
        self.save_hyperparameters()

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = OrangesDataset(
                self.data_dir, self.df_train,
                transform=self.transform_train
            )

            self.val_dataset = OrangesDataset(
                self.data_dir, self.df_validation,
                transform=self.transform_test
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = OrangesDataset(
                self.data_dir, self.df_test,
                transform=self.transform_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )