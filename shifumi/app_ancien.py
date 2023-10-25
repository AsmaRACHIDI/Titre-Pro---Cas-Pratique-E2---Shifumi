import os
import random
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Chargement du modèle pré-entraîné
model = tf.keras.models.load_model('model/MobilnetV2-shifumi.h5')
labels = ['Pierre', 'Feuille', 'Ciseaux']

# Variables pour le jeu
player_gesture = None
computer_gesture = None
result = None
webcam_enabled = False  # Indique si la reconnaissance webcam est activée

# Fonction pour effectuer la prédiction sur un frame vidéo
def predict_gesture(frame):
    # Prétraitement de l'image (redimensionnement et normalisation)
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)  # Ajouter une dimension batch

    # Prédiction du geste
    prediction = model.predict(frame)
    predicted_class = np.argmax(prediction)

    return labels[predicted_class]

# Fonction pour jouer un tour du jeu
def play_turn(player_choice):
    global player_gesture, computer_gesture, result
    player_gesture = player_choice
    computer_gesture = random.choice(labels)

    if player_gesture == computer_gesture:
        result = 'Égalité'
    elif (player_gesture == 'Pierre' and computer_gesture == 'Ciseaux') or \
         (player_gesture == 'Feuille' and computer_gesture == 'Roche') or \
         (player_gesture == 'Ciseaux' and computer_gesture == 'Papier'):
        result = 'Gagné'
    else:
        result = 'Perdu'

# Fonction pour capturer la vidéo de la webcam
def webcam_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if webcam_enabled:
            gesture = predict_gesture(frame)
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Affichez le résultat du tour
        if player_gesture is not None:
            cv2.putText(frame, f'Joueur : {player_gesture}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Ordinateur : {computer_gesture}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Résultat : {result}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html', webcam_enabled=webcam_enabled)

# Route pour activer/désactiver la reconnaissance webcam
@app.route('/toggle_webcam', methods=['POST'])
def toggle_webcam():
    global webcam_enabled
    webcam_enabled = not webcam_enabled
    return '', 204  # Réponse sans contenu

# Route pour le flux vidéo de la webcam
@app.route('/video_feed')
def video_feed():
    return Response(webcam_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route pour le choix du joueur
@app.route('/choose', methods=['POST'])
def choose():
    if not webcam_enabled:
        player_choice = request.form['choice']
        play_turn(player_choice)
    return '', 204  # Réponse sans contenu

# Route pour obtenir le résultat du jeu
@app.route('/result')
def get_result():
    global result
    return result

if __name__ == '__main__':
    app.run(debug=True)
