from model import MyModel
from preprocessing import DataPreprocess

# Declare constants
img_size = 254
nb_channels = 3
batch_size = 8
validation_split = 0.2
train_dir = "Data/Training/"
test_dir = "Data/Test/"
train_refined_dir = "Data/Training_refined/"
best_model_path = "Model/tl_best_model_22apr.h5"
num_trainable_layers = 102
epochs = 100

# # load objects
data_obj = DataPreprocess(img_size, nb_channels, batch_size, validation_split)
model_obj = MyModel()

# Data Pre-processing
data_obj.refine_train_data(train_dir, train_refined_dir)
train_gen, val_gen, test_gen = data_obj.image_augmentation(train_refined_dir, test_dir)

# View preprocessed samples
imgs, labels = next(train_gen)
data_obj.view_imagegen_samples(imgs)
print(labels)

# Load pretrained model and Create fine tuned Model
my_model = model_obj.load_and_finetune_model(img_size, nb_channels, num_trainable_layers)

# Verify trainable layers
count = 0
for layer in my_model.layers:
    if layer.trainable:
        count += 1
print(count)

# # train custom model
history = model_obj.train_model(my_model, train_gen, val_gen, epochs)
model_obj.plot_history(history)

# # load and predict
pred = model_obj.load_and_predict(best_model_path, test_gen)

# Evaluate model
labels = test_gen.classes
cm, cr = model_obj.evaluate_model(pred, labels)
print(cm, cr)
