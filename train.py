import copy

import torch
import torchvision.models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn, optim

if __name__ == '__main__':
    datas_train = datasets.flowers102.Flowers102(
        root='.',
        split="train",
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )
    datas_val = datasets.flowers102.Flowers102(
        root='.',
        split="val",
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )
    datas_test = datasets.flowers102.Flowers102(
        root='.',
        split="test",
        download=True,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    )

    batch_size = 16
    train_dataloader = DataLoader(datas_train, batch_size=batch_size, drop_last=True)
    val_dataloader = DataLoader(datas_val, batch_size=batch_size, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(datas_test, batch_size=batch_size, drop_last=True)

    # ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # freeze paras
    for para in model.parameters():
        para.requires_grad = False

    # new fc layer
    model.fc = nn.Linear(model.fc.in_features, 102)
    # loss function, optim and sched.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using {device} device to train model")

    # training
    model.to(device)
    num_epochs = 25
    best_weights = None
    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        # training
        epoch_cnt = 0
        epoch_loss = 0.
        epoch_corrects = 0
        model.train()
        
        for batch, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # compute prediction error
            pred = model(images)
            _, preds = torch.max(pred, 1)
            loss = loss_fn(pred, labels)
            
            # backpropagation
            loss.backward()
            optimizer.step()
            

            # stat
            batch_avg_loss, current = loss.item(), (batch + 1) * (len(images))
            acc = torch.sum(preds == labels.data) / len(images)
            epoch_corrects += torch.sum(preds == labels.data)
            epoch_cnt += images.size(0)
            epoch_loss += batch_avg_loss * images.size(0)
            print(f"training: loss:{batch_avg_loss:>7f}  acc:{acc:>7f}  [{current:>5d}/{len(train_dataloader.dataset)}]")

        epoch_acc = epoch_corrects / epoch_cnt 
        print(f"train loss:{epoch_loss/epoch_cnt:.5f} acc:{epoch_acc:.5f}")

        # validating
        best_val_acc = 0.
        total_val_cnt = 0
        total_accuracy = 0.
        total_val_loss = 0.
        model.eval()
        for batch, (images, labels) in enumerate(val_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                pred = model(images)
                _, preds = torch.max(pred, 1)
                loss = loss_fn(pred, labels)

            # stat
            current = (batch + 1) * (len(images))
            acc = torch.sum(preds == labels.data) / len(images)
            total_val_cnt += images.size(0)
            total_val_loss += loss.item() * images.size(0)
            total_accuracy += torch.sum(preds == labels.data)
            #print(f"validating: loss:{loss.item():>7f}  acc:{acc:>7f}  [{current:>5d}/{len(val_dataloader.dataset)}]")

        epoch_avg_val_loss = total_val_loss / total_val_cnt
        epoch_avg_acc = total_accuracy / total_val_cnt
        print(f"val loss:{epoch_avg_val_loss:.5f} acc:{epoch_avg_acc:.5f}")

        # save best model
        if epoch_avg_acc >= best_val_acc:
            best_val_acc = epoch_avg_acc
            best_weights = copy.deepcopy(model.state_dict())

    # test
    model.load_state_dict(best_weights)
    model.eval()
    corrects = 0.
    total = 0
    for batch, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        _, cls = torch.max(preds, 1)
        correct_num = torch.sum(cls == labels.data)
        corrects += correct_num
        total += len(images)
        acc = correct_num / len(images)
        print(f"test-{batch}: acc:{acc:>7f}  [{total:>5d}/{len(test_dataloader.dataset)}]")

    test_acc = corrects / total
    print(f"test_total acc:{test_acc:>7f}")
    # save best model
    torch.save(model, 'resnet18.pth')
