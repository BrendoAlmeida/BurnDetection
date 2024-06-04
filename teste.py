import os
import cv2
import numpy as np
from scipy.stats import entropy

# Função para calcular a entropia de Shannon de uma imagem
def shannon_entropy(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    hist = hist / hist.sum()
    return entropy(hist, base=2)

# Função para detectar fogo e fumaça em uma imagem
def detectar_fogo_fumaca(frame):
    # Converter a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir intervalos de cor para a fumaça
    lowerFumaca = np.array([0, 100, 100])
    upperFumaca = np.array([13, 255, 255])
    mascaraFumaca = cv2.inRange(hsv, lowerFumaca, upperFumaca)

    # Definir intervalos de cor para o centro do fogo (amarelo)
    lowerCentroFogo = np.array([20, 100, 100])
    upperCentroFogo = np.array([30, 255, 255])
    mascaraCentroFogo = cv2.inRange(hsv, lowerCentroFogo, upperCentroFogo)

    # Definir intervalos de cor para o centro do fogo (branco)
    lowerCentroFogoBranco = np.array([0, 0, 200])
    upperCentroFogoBranco = np.array([180, 55, 255])
    mascaraCentroFogoBranco = cv2.inRange(hsv, lowerCentroFogoBranco, upperCentroFogoBranco)

    # Combinar todas as máscaras
    mascara = cv2.bitwise_or(mascaraFumaca, mascaraCentroFogo)
    mascara = cv2.bitwise_or(mascara, mascaraCentroFogoBranco)

    # Encontrar contornos na máscara combinada
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtro de altura para ignorar o céu (assumindo que o céu está na parte superior)
    altura_frame = frame.shape[0]
    limite_altura_ceu = int(altura_frame * 0.3)  # Ajustar conforme necessário

    maior_contorno = None
    maior_area = 0

    for contour in contours:
        # Ignorar contornos pequenos e contornos na parte superior da imagem
        if cv2.contourArea(contour) < 500:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        if y < limite_altura_ceu:
            continue

        # Recortar a região do contorno da imagem original para calcular a entropia
        roi = frame[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ent = shannon_entropy(gray_roi)

        # Considerar um contorno como fogo se a entropia for alta o suficiente
        if ent > 0:  # Ajuste este valor conforme necessário
            if cv2.contourArea(contour) > maior_area:
                maior_area = cv2.contourArea(contour)
                maior_contorno = contour

    # Desenhar o maior contorno que foi considerado fogo
    if maior_contorno is not None:
        (x, y, w, h) = cv2.boundingRect(maior_contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(frame, [maior_contorno], -1, (0, 255, 255), 2)

        # Encontrar o centro do contorno
        M = cv2.moments(maior_contorno)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Desenhar um círculo no centro do contorno
        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
        cv2.putText(frame, "Centro", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame


for img in os.listdir("images/queimada"):
    # Detectar fogo na imagem
    frame = detectar_fogo_fumaca(cv2.imread(f"images/queimada/{img}"))

    # Mostrar a imagem com os contornos do fogo
    cv2.imshow('Detecção de Fogo', frame)
    cv2.waitKey(0)

    # Sair do loop ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
