from utils import *


def test():
        
    # Loading Pre-trained Model and pickled fiiles 
    try:
        with open(str(pickel_files_path)+'/words.pkl', 'rb') as file:
            words = pickle.load(file)

        # Load classes
        with open(str(pickel_files_path)+'/classes.pkl', 'rb') as file:
            classes = pickle.load(file)
        # print("tring loading model")
        model = keras.models.load_model(str(model_path)+'/Medical-chatbot.h5')
        # print("Model Loaded Successfully")
      

    except Exception as e:
        traceback.print_exc()
        
    return model,words,classes

test()