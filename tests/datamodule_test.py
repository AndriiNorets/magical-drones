from magical_drones.datasets.magmap import MagMapV1
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomCrop, Resize, ToTensor

if __name__ == "__main__":
    data_link = "czarna-magia/mag-map"
    batch_size = 32

    train_transform = transforms.Compose([
        RandomCrop(size=(224, 224)),
        RandomHorizontalFlip(p=0.5),
        ToTensor()  
    ])

    test_transform = transforms.Compose([
        Resize(size=(224, 224)),
        ToTensor()  
    ])
    
    magmap = MagMapV1(data_link, 
                      batch_size=batch_size, 
                      train_transform=train_transform, 
                      test_transform=test_transform)

    print("Preparing data...")
    magmap.prepare_data()

    print("Setting up datasets...")
    magmap.setup()

    print("Testing train_dataloader...")
    train_loader = magmap.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break

    print("Testing val_dataloader...")
    val_loader = magmap.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break

    print("Testing test_dataloader...")
    test_loader = magmap.test_dataloader()
    for i, batch in enumerate(test_loader):
        print(f"Batch {i + 1}: {batch}")
        if i == 2:
            break
