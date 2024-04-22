import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import DINOLinearClassifier
from load_data import *
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
# from transformers import Trainer, TrainingArguments
import torch
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import torch.optim as optim
import torchvision


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# def forward_pass_with_label(batch):
#     # Place all input tensors on the same device as the model
#     inputs = {k:v.to(device) for k,v in batch.items()
#               if k in tokenizer.model_input_names}

#     with torch.no_grad():
#         output = DINOLinearClassifier(**inputs)
#         pred_label = torch.argmax(output.logits, axis=-1)
#         loss = cross_entropy(output.logits, batch["label"].to(device),
#                              reduction="none")
#     # Place outputs on CPU for compatibility with other dataset columns
#     return {"loss": loss.cpu().numpy(),
#             "predicted_label": pred_label.cpu().numpy()}

if __name__ == '__main__':
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOLinearClassifier(num_classes=2)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, 2) # 2 classes

    # loss 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    data_transforms = {
        "train": 
            torchvision.transforms.Compose([
                transforms.Resize(size=(196,196), interpolation=transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        "validation": 
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((196, 196)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    }

    print("Loading data...")
    train_dataset = GeoPACHAImageDataset(data_file="/workspace/geopacha/jn/Train.csv",
                                transform=data_transforms["train"])
    validation_dataset = GeoPACHAImageDataset(data_file="/workspace/geopacha/jn/Train.csv",
                                transform=data_transforms["validation"])

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=4),
        "validation": torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=4)
    }

    trainiter = iter(dataloaders["train"])
    validationiter = iter(dataloaders["validation"])

    logging_steps = len(train_dataset) // batch_size 
    model_name = f"dinov2_vitl14-finetuned" 

    from dataclasses import dataclass

    @dataclass
    class TrainingConfig:
        epochs: int = 3 # TODO
        batch_size: int = 1024
        learning_rate: float = 1e-4
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = TrainingConfig(epochs=2, batch_size=batch_size, learning_rate=2e-5)

    # training_args = TrainingArguments(output_dir="./output",
    #                                 #output_dir=model_name,
    #                                 num_train_epochs=2,
    #                                 learning_rate=2e-5,
    #                                 per_device_train_batch_size=batch_size,
    #                                 per_device_eval_batch_size=batch_size,
    #                                 weight_decay=0.01,
    #                                 evaluation_strategy="epoch",
    #                                 disable_tqdm=False,
    #                                 logging_steps=logging_steps,
    #                                 # push_to_hub=True,
    #                                 log_level="error")


    # trainer = Trainer(model=model, args=training_args,
    #                 compute_metrics=compute_metrics,
    #                 train_dataset=dataset["train"],
    #                 eval_dataset=dataset["validation"]) # TODO 
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print("Training...")                
    # Training and validation loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for batch in dataloaders["train"]:
            inputs, labels = batch
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        model.eval()
        val_loss = 0
        val_accuracy = []
        with torch.no_grad():
            for batch in dataloaders["validation"]:
                inputs, labels = batch
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = cross_entropy(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                predictions = outputs.argmax(dim=1, keepdim=True)
                correct = predictions.eq(labels.view_as(predictions)).sum().item()
                val_accuracy.append(correct / len(labels))

        # Calculate average losses and accuracy
        avg_train_loss = train_loss / len(dataloaders["train"])
        avg_val_loss = val_loss / len(dataloaders["validation"])
        avg_val_accuracy = sum(val_accuracy) / len(val_accuracy)

        print(f'Epoch {epoch+1}/{config.epochs}, Train Loss: {avg_train_loss:.4f}, '
            f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}')

    print("Training complete!")

    def plot_confusion_matrix(y_preds, y_true, labels):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title("Normalized confusion matrix")
        plt.show()

    # preds_output = trainer.predict(dataset["validation"])
    # preds_output.metrics
    # y_preds = np.argmax(preds_out_size
    # put.predictions, axis=1)
    # plot_confusion_matrix(y_preds, y_valid, labels)

    # # Convert dataset back to PyTorch tensors
    # dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    # # Compute loss values
    # dataset["validation"] = dataset["validation"].map(
    #     forward_pass_with_label, batched=True, batch_size=16)

    # dataset.set_format("pandas")
    # cols = ["text", "label", "predicted_label", "loss"]
    # df_test = dataset["validation"][:][cols]

    # def label_int2str(row):
    #     return df_test["train"].features["label"].int2str(row)

    # df_test["label"] = df_test["label"].apply(label_int2str)
    # df_test["predicted_label"] = (df_test["predicted_label"]
    #                             .apply(label_int2str))
    # df_test.sort_values("loss", ascending=True).head(10)

    # save_model_path = "dinov2_vitl14-finetuned.pth"
    # torch.save(model.state_dict(), save_model_path)
