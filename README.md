## Aimonk Multi-Label Image Classification

## Problem

This project solves a multi-label image classification problem.

Each image contains clothing items and may have multiple attributes (Attr1–Attr4).

The labels file format:

~~~
image_name  Attr1  Attr2  Attr3  Attr4
~~~

Where:

- 1 → attribute present

- 0 → attribute absent

- NA → attribute information missing

## Approach

- Model

  -  Used ResNet50 pretrained on ImageNet

  -  Fine-tuned only the final layer

  -  Output layer modified to 4 neurons (one per attribute)

  -  Used Sigmoid activation for multi-label prediction

- Loss Function

  -  Used BCEWithLogitsLoss

  -  Applied class-wise positive weights to handle imbalance

  -  Implemented masked loss to ignore NA values

- Data Processing

  -  Image resizing to 224×224

  -  ImageNet normalization

  -  Random horizontal flip (augmentation)

## Handling Special Cases
- Missing Labels (NA)

  -  Converted NA → -1

  -  Ignored missing attributes during loss calculation

  -  Ensured images were not discarded

- Class Imbalance

  -  Computed:

      ~~~
      pos_weight = negative_samples / positive_samples
      ~~~

  -  Passed into loss function to balance rare attributes
 
## Output
- Training Code

  -  train.py → trains model and saves model.pth

- Loss Curve

  -  X-axis → iteration_number

  -  Y-axis → training_loss

  -  Title → Aimonk_multilabel_problem

- Inference Code

  -  inference.py

  -  Takes image as input

  -  Prints predicted attributes

Example:
~~~
Attributes present: ['Attr1', 'Attr4']
~~~

## Project Structure

dataset.py       → Custom dataset loader

data_loader.py   → DataLoader + transforms

model.py         → Pretrained ResNet50 modification

loss.py          → Masked BCE + imbalance handling

train.py         → Training script

inference.py     → Prediction script

model.pth        → Saved model weights

loss_curve.png   → Training loss plot

## Possible Improvements (If More Time Available)

- Add validation split

- Calculate F1-score / mAP

- Use different models:

  -  EfficientNet

  -  MobileNet

  -  DenseNet

- Unfreeze last ResNet block for deeper fine-tuning

- Use learning rate scheduler

- Try Focal Loss for extreme imbalance

- Tune decision threshold instead of fixed 0.5

- Add stronger augmentations (ColorJitter, RandomCrop)

## Summary

This project demonstrates:

  -  Multi-label classification

  -  Transfer learning

  -  Handling missing labels

  -  Handling imbalanced dataset

  -  Clean modular deep learning pipeline

## Author

Ashish Mittal
