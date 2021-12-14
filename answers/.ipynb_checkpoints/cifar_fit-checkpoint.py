model.fit(
    train_generator,
    verbose=1,
    epochs=10,
    steps_per_epoch=10,
    validation_data=test_generator,
    validation_steps=10,
)
