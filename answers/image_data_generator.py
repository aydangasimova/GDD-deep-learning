train_iterator = train_data_generator.flow_from_directory(
    directory=os.path.join(temp_dir, "train"),
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)
valid_iterator = valid_data_generator.flow_from_directory(
    directory=os.path.join(temp_dir, "valid"),
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=8,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)
test_iterator = test_data_generator.flow_from_directory(
    directory=os.path.join(temp_dir, "test"),
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
    seed=42,
)
