import data
from sklearn.model_selection import train_test_split

def train(model, filename, track1, side_cameras):
    samples = data.load_data(track1, side_cameras)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    train_generator = data.generator(train_samples, batch_size=32)
    validation_generator = data.generator(validation_samples, batch_size=32)

    model.fit_generator(train_generator, samples_per_epoch = len(train_samples), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)
    if track1:
        if side_cameras:
            name = filename + '_track1_sides'
        else:
            name = filename + '_track1'
    else:
        if side_cameras:
            name = filename + '_all_sides'
        else:
            name = filename + '_all'

    model.save(name)