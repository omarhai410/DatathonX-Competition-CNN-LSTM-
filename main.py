import io
import joblib
from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from mtranslate import translate
import pickle
import tensorflow as tf
import base64

app = Flask(__name__)

# Load your trained model and tokenizer
with open("image_captioning_model.pkl", 'rb') as model_file:
    caption_model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the InceptionV3 model
inception_v3_model = tf.keras.applications.InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
inception_v3_model.layers.pop()
inception_v3_model = tf.keras.Model(inputs=inception_v3_model.inputs, outputs=inception_v3_model.layers[-2].output)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

# ...

def beam_search_generator(image_features, K_beams=3, log=False):
    start = [tokenizer.word_index['start']]

    start_word = [[start, 0.0]]

    for _ in range(35):  # Assuming max caption length is 15
        temp = []
        for s in start_word:
            sequence = tf.keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=34).reshape((1, 34))

            preds = caption_model.predict([image_features.reshape(1, 2048), sequence], verbose=0)

            if isinstance(preds[0], (np.generic, np.ndarray)):
                word_preds = np.argsort(preds[0])[-K_beams:]
            else:
                word_preds = [preds[0]]

            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w]) if isinstance(preds[0], (np.generic, np.ndarray)) else np.log(w)
                else:
                    prob += preds[0][w] if isinstance(preds[0], (np.generic, np.ndarray)) else w
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-K_beams:]

    # Check if start_word is scalar, if so, convert it to a list
    if isinstance(start_word, (np.generic, np.ndarray)):
        start_word = [start_word]

    start_word = start_word[-1][0]
    captions_ = [tokenizer.index_word[i] for i in start_word]

    final_caption = []

    for i in captions_:
        if i != 'end':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    confidence = start_word[-1][1] if isinstance(start_word[-1], list) else start_word[1]

    return final_caption, confidence


# Route for the home page
@app.route('/')
def home():
    return render_template('img.html')


# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        image_path = 'uploads/' + secure_filename(file.filename)
        file.save(image_path)

        img_array = preprocess_image(image_path)
        features = inception_v3_model.predict(img_array)
        prediction, probabilities = beam_search_generator(features)

        # Translate the prediction to Arabic
        translated_prediction = translate(prediction, 'ar')

        # Open the image using Pillow
        img = Image.open(image_path)

        # Convert the image to bytes and then encode it in base64
        img_bytes = io.BytesIO()
        # Convert the image to 'RGB' mode before saving as JPEG
        rgb_img = img.convert('RGB')
        rgb_img.save(img_bytes, format="JPEG")
        img_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')


        return render_template('img.html', image_data=img_data, prediction=prediction,
                           translated_prediction=translated_prediction, image_probabilities=probabilities)

# Charger le modèle pré-entraîné
modele_preentraine = joblib.load('naive_bayes_countvectorizer_3gram_model.joblib')

# Charger le vectorizer
vectorizer = joblib.load('count_vectorizer.joblib')  # Assurez-vous de sauvegarder votre vectorizer pendant l'entraînement


@app.route('/prediction2', methods=['POST'])
def effectuer_prediction2():
    if request.method == 'POST':
        texte_input = request.form.get('texte', '')
        # Vectoriser le texte d'entrée en utilisant le même vectorizer utilisé pendant l'entraînement
        texte_vectorise = vectorizer.transform([texte_input])
        # Faire une prédiction en utilisant le modèle pré-entraîné avec la variable renommée à prediction2
        prediction2_resultat = modele_preentraine.predict(texte_vectorise)[0]

        # Translate the text prediction to Arabic
        if prediction2_resultat == 'POS':
            text_prediction = translate('Positive', 'ar')
            return render_template('img.html', text_prediction=text_prediction, texte=texte_input, afficher_modal='positif')
        elif prediction2_resultat == 'NEG':
            text_prediction = translate('Negative', 'ar')
            return render_template('img.html', text_prediction=text_prediction, texte=texte_input, afficher_modal='negatif')
        else:
            text_prediction = translate('Neutral', 'ar')
            return render_template('img.html', text_prediction=text_prediction, texte=texte_input, afficher_modal='neutre')
if __name__ == '__main__':
    app.run(debug=True)
