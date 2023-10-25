# Importez les bibliothèques nécessaires
from flask import Flask, request, jsonify, render_template, Response
import tensorflow as tf
import cv2
import base64
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Charger le modèle IA MobileNetV2
model = load_model('model/MobilnetV2-shifumi.h5')

# La webcam ne sera pas activée par défaut pour éviter des problèmes
# Vous pouvez ajouter un bouton "Activer la webcam" sur votre page HTML

cap = None  # La webcam sera initialisée plus tard

# Définir les classes du jeu (Pierre, Feuille, Ciseaux)
classes = ['Pierre', 'Feuille', 'Ciseaux']
gesture_names = {0: 'Pierre', 1: 'Feuille', 2: 'Ciseaux'}

# Scores du jeu
user_score = 0
computer_score = 0

# Fonction pour jouer au jeu
def play_game(user_choice, computer_choice):
    global user_score, computer_score
    if user_choice == computer_choice:
        return "Égalité"
    elif (
        (user_choice == "Pierre" and computer_choice == "Ciseaux")
        or (user_choice == "Feuille" and computer_choice == "Pierre")
        or (user_choice == "Ciseaux" and computer_choice == "Feuille")
    ):
        user_score += 1
        return "Vous gagnez !"
    else:
        computer_score += 1
        return "L'ordinateur gagne."
    
def preprocess_image(frame):
    # Redimensionnez l'image à la taille attendue (224x224)
    frame = cv2.resize(frame, (224, 224))
    # Normalisez les pixels de l'image
    frame = frame / 255.0
    return frame

# Fonction pour prédire le choix de l'utilisateur
def gen_frames():
    global cap  # Utilisez la variable globale cap pour accéder à la webcam

    if cap is None:
        # Initialisez la webcam ici (vérifiez si elle est ouverte avec succès)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return Response("Aucune caméra disponible.", mimetype='text/plain')

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Prétraitez l'image capturée
            processed_frame = preprocess_image(frame)

            # Faites une prédiction avec le modèle
            prediction = model.predict(np.expand_dims(processed_frame, axis=0))

            # Obtenez la classe prédite
            predicted_class = np.argmax(prediction)

            # Affichez le résultat sur la frame
            text = gesture_names[predicted_class]
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route pour la page d'accueil
@app.route('/')
def home():
    global user_score, computer_score
    user_score = 0
    computer_score = 0
    return render_template('index.html')

# Route pour jouer avec les boutons
@app.route('/play', methods=['POST'])
def play_game_buttons():
    global user_score
    global computer_score

    try:
        data = request.get_json()
        user_choice = data['choice']

        # Générer le choix de l'ordinateur (aléatoire, à vous de choisir la logique)
        computer_choice = np.random.choice(classes)

        # Jouer au jeu en utilisant les choix
        game_result = play_game(user_choice, computer_choice)

        response = {
            'user_choice': user_choice,
            'computer_choice': computer_choice,
            'result': game_result,
            'user_score': user_score,
            'computer_score': computer_score
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# Route pour la page avec les boutons
@app.route('/play_buttons')
def play_buttons():
    return render_template('play_buttons.html')

# Route pour réinitialiser les scores
@app.route('/reset', methods=['POST'])
def reset_scores():
    global user_score, computer_score
    user_score = 0
    computer_score = 0
    return jsonify({'message': 'Scores réinitialisés avec succès.'})

# Ajoutez cette route pour jouer avec la webcam
@app.route('/play_webcam')
def play_with_cam():
    return render_template('play_webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_json()
    image_base64 = image_data.get('image')

    # Convertissez l'image base64 en une image OpenCV
    image_bytes = base64.b64decode(image_base64.split(',')[1])
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Prétraitez l'image
    processed_frame = preprocess_image(image)

    # Faites la prédiction
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = np.argmax(prediction)
    predicted_gesture = gesture_names[predicted_class]

    return jsonify({'prediction': predicted_gesture})

if __name__ == '__main__':
    app.run(debug=True)
