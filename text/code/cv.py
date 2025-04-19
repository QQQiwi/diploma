import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import ViTForImageClassification, AdamW, get_scheduler
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from transformers import ViTFeatureExtractor
import PIL
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import numpy as np
import timm
import pickle


def build_conf_matrix(labels, predict, class_name):
    lab, pred = [], []
    for i in range(len(labels)):
        if predict[i] == class_name:
            pred.append(0)
        else:
            pred.append(1)
        if labels[i] == class_name:
            lab.append(0)
        else:
            lab.append(1)
    return confusion_matrix(lab, pred, normalize='true')


def vit_experiment(data_dir):
    classes = os.listdir(data_dir)
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-384")
    imgs = []
    lbls = []
    cnt = 0
    class2lbl = {"AKR": 1, "not_AKR": 0}
    for cl in classes:
        class_path = data_dir + '/' + cl
        images = os.listdir(class_path)
        for image in images:
            try:
                input = PIL.Image.open(class_path + '/' + image).convert("RGB")
                imgs.append(feature_extractor(input, return_tensors="pt")['pixel_values'][0])
                lbls.append(class2lbl[cl])
            except:
                cnt += 1

    train_images, val_images, train_labels, val_labels = train_test_split(imgs, lbls, test_size=.2, stratify=lbls)
    
    class AKRDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {}
            item['inputs'] = self.images[idx]
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
        
    train_data = AKRDataset(train_images, train_labels)
    val_data = AKRDataset(val_images, val_labels)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True)


    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-384", num_labels=2, ignore_mismatched_sizes=True)

    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    num_epoch = 5

    losses_for_train = []
    losses_for_val = []
    num_training_steps = num_epoch * len(train_loader)
    lr_scheduler = get_scheduler('linear',
                                    optimizer=optim,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps)

    for epoch in range(num_epoch):
        loop = tqdm(train_loader, leave=True)
        model.train()
        for batch in loop:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            loss.backward()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            optim.step()
            losses_for_train.append(loss.item())
            lr_scheduler.step()
            optim.zero_grad()
        model.eval()

        for batch in val_loader:
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                outputs = model(inputs, labels=labels)
            loss = outputs[0]
            losses_for_val.append(loss.item())


    train_loss = []
    val_loss = []
    tl = losses_for_train[0]
    for i in range(1, len(losses_for_train)):
        if i % (len(train_loader) - 1):
            tl+=losses_for_train[i]
        else:
            train_loss.append(tl / len(train_loader))
            tl = 0
    tl = losses_for_val[0]
    for i in range(1, len(losses_for_val)):
        if i % (len(val_loader) - 1):
            tl+=losses_for_val[i]
        else:
            val_loss.append(tl / len(val_loader))
            tl = 0


    all_predictions = []
    all_references = []
    for batch in val_loader:
        inputs = batch['inputs'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim= -1)
        all_predictions.extend(predictions.cpu().numpy())
        all_references.extend(batch['labels'].cpu().numpy())

    print('accuracy:{}'.format(accuracy_score(all_predictions, all_references)))
    print('precision:{}'.format(precision_score(all_predictions, all_references ,average='weighted')))
    print('recall:{}'.format(recall_score(all_predictions, all_references, average='weighted')))
    print('f1-score:{}'.format(f1_score(all_predictions, all_references, average='weighted')))
    print(build_conf_matrix(all_references, all_predictions, 1))
    model.save_pretrained(f"vit_{data_dir}.pth")


def load_split_train_test(datadir, valid_size = 0.2):
    train_trainsforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    train_data = datasets.ImageFolder(datadir,transform=train_trainsforms)

    num_train = len(train_data)                               
    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train)) 
    np.random.shuffle(indices) 

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    print(len(train_idx), len(test_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler  = SubsetRandomSampler(test_idx)

    train_loader = DataLoader(train_data,sampler=train_sampler,batch_size=64)
    test_loader = DataLoader(train_data,sampler=test_sampler,batch_size=64)
    return train_loader,test_loader


def resnet34_experiment(data_dir):
    train_loader,test_loader = load_split_train_test(data_dir, 0.2)

    if torch.cuda.is_available():
        device = torch.device("cuda")  # здесь вы можете продолжить, например,cuda:1 cuda:2... и т. д. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    model = models.resnet34(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(nn.Linear(512,512),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(512,7),
                            nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(),lr=0.0001)
    model.to(device)

    epochs = 10
    steps = 0
    running_loss = 0
    train_losses, test_losses = [],[]

    for epoch in tqdm.tqdm(range(epochs)):
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            steps +=1

        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs,labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out2 = model(inputs)
                batch_loss = criterion(out2,labels)
                test_loss +=batch_loss.item()

                ps = torch.exp(out2)
                top_pred, top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print(f"Epoch {epoch+1}/{epochs}"
                    f"Train loss: {running_loss/len(train_loader):.3f}",
                    f"Test loss: {test_loss/len(test_loader):.3f} "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}")
        running_loss = 0
        model.train()

    l = []
    c = []
    model.eval()
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out2 = model(inputs)

            ps = torch.exp(out2)
            top_pred, top_class = ps.topk(1,dim=1)
            for i in range(len(top_class)):
                l.append(labels[i].item())
                c.append(top_class[i].item())

    with open(f"resnet34_{data_dir}.pth", 'wb') as file:
        pickle.dump(model, file)

    print(build_conf_matrix(l, c, 1))
    print('accuracy:{}'.format(accuracy_score(l, c)))
    print('precision:{}'.format(precision_score(l, c,average='weighted')))
    print('recall:{}'.format(recall_score(l, c,average='weighted')))
    print('f1-score:{}'.format(f1_score(l, c,average='weighted')))


def xception_experiment(data_dir):
    model = timm.create_model('xception', pretrained=True, num_classes=2)
    model.eval()
    train_loader, test_loader = load_split_train_test(data_dir, 0.2)
    if torch.cuda.is_available():
        device = torch.device("cuda")  # здесь вы можете продолжить, например,cuda:1 cuda:2... и т. д. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    model.to(device)

    epochs = 5
    steps = 0
    running_loss = 0
    train_losses,test_losses = [],[]


    for epoch in tqdm.tqdm(range(epochs)):
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            out = model(inputs)
            log_out = torch.nn.functional.log_softmax(out)
            loss = criterion(log_out,labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            steps +=1

        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs,labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                out2 = model(inputs)
                log_out2 = torch.nn.functional.log_softmax(out2)
                batch_loss = criterion(log_out2,labels)
                test_loss += batch_loss.item()

                ps = torch.exp(out2)
                top_pred,top_class = ps.topk(1,dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))

        print(f"Epoch {epoch+1}/{epochs}"
                    f"Train loss: {running_loss/len(train_loader):.3f}",
                    f"Test loss: {test_loss/len(test_loader):.3f} "
                    f"Test accuracy: {accuracy/len(test_loader):.3f}")
        running_loss = 0
        model.train()

    l = []
    c = []
    model.eval()
    with torch.no_grad():
        for inputs,labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out2 = model(inputs)

            ps = torch.exp(out2)
            top_pred, top_class = ps.topk(1,dim=1)
            for i in range(len(top_class)):
                l.append(labels[i].item())
                c.append(top_class[i].item())
    
    with open(f"xception_{data_dir}.pth", 'wb') as file:
        pickle.dump(model, file)

    print(build_conf_matrix(l, c, 1))
    print('accuracy:{}'.format(accuracy_score(l, c)))
    print('precision:{}'.format(precision_score(l, c,average='weighted')))
    print('recall:{}'.format(recall_score(l, c,average='weighted')))
    print('f1-score:{}'.format(f1_score(l, c,average='weighted')))


def main():
    ethalon_block_sizes = [4, 16, 32]
    for bs in ethalon_block_sizes:
        data_dir = f'cv_classification_{bs}'
        vit_experiment(data_dir)
        resnet34_experiment(data_dir)
        xception_experiment(data_dir)


if __name__ == "__main__":
    main()