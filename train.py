import data
from sklearn.model_selection import train_test_split

def train(model, filename, track1, side_cameras, use_generators,epochs):
    samples = data.load_data(track1, side_cameras)
    
    if use_generators:

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
        train_generator = data.generator(train_samples, batch_size = 100)
        validation_generator = data.generator(validation_samples, batch_size = 100)
    
        model.fit_generator(train_generator, samples_per_epoch = 5*len(train_samples)/100, 
                        validation_data=validation_generator,
                        nb_val_samples=5*len(validation_samples)/100, nb_epoch=5)
    else:
        print("NO GENERATORS")
        X_data, y_data = data.load_images(samples)
        model.fit(X_data, y_data, validation_split=0.2, shuffle=True, epochs=epochs)
    # print last layer
    print( model.layers[-1].get_weights() )

    #save model
    if track1:
        if side_cameras:
            name = filename + '_track1_sides.h5'
        else:
            name = filename + '_track1.h5'
    else:
        if side_cameras:
            name = filename + '_all_sides.h5'
        else:
            name = filename + '_all.h5'


    model.save(name)