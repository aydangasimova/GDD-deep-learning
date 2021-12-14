batch_size = 50

train_generator = train_datagen.flow_from_directory(
    directory="../output/CIFAR10/train/",
    target_size=(128, 128),
    shuffle=True,
    batch_size=batch_size,
)
test_generator = train_datagen.flow_from_directory(
    directory="../output/CIFAR10/test/",
    target_size=(128, 128),
    shuffle=True,
    batch_size=batch_size,
)
