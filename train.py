from utils1 import *

from utils1 import load_data, organize_Data, generate_dataset, train_and_save_model, build_model

def model_train():
    # Data Preprocessing
    nltk.download('punkt')
    from nltk.stem.lancaster import LancasterStemmer

    # Loading Dataset
    intents = load_data(r'./Dataset/intents.json')

    # Organizing Dataset
    words, classes, documents = organize_Data(intents)

    # Generating training Data
    train_x, train_y = generate_dataset(words, classes, documents)

    # Saving training Data as pkl files for later usage
    train_and_save_model(pickel_files_path, words, classes, train_x, train_y)

    num_folds = 2
    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in kf.split(train_x):
        train_x_fold, test_x_fold = np.array(train_x)[train_index], np.array(train_x)[test_index]
        train_y_fold, test_y_fold = np.array(train_y)[train_index], np.array(train_y)[test_index]

        model = build_model(train_x, train_y)
        model.fit(train_x_fold, train_y_fold, epochs=200, batch_size=8, verbose=1)

    # Save the model
    model.save('./models/Medical-chatbot.h5')

    # End of Model Training

    return model

trained_model = model_train()