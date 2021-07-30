# Transfer Learning for Stanford Cars datset
VGG16 Transfer Learning for classification of the Stanford Cars dataset from Kaggle
#
First, I renamed the train and test images for simplicity. Then the images were preprocessed using `preprocess.py`. The preprocessed datasets were converted and saved to `.npy` files for less overhead loading time for model training.

For model training, VGG16 (trained on ImageNet) was used. `train_model.py` will generate and save the model, loss/accuracy vs epochs plot, and the model architecture.

![loss_acc_plot](https://user-images.githubusercontent.com/60960803/127654731-147ffc7f-d3b2-4b0e-bf5e-f805abc9d9c4.jpg)

The loss/accuracy vs epochs plot shows that the model overfit. No fine-tuning was done afterwards as the purpose of this project was just to familiarize with transfer learning using Keras.

Running `predict.py` will evaluate the model using the provided test set. It gave 81% accuracy.
#
If you have any questions or suggestions, feel free to contact me here. Thanks for reading!
