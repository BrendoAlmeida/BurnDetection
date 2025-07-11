import cv2
import numpy as np
from scipy.stats import entropy, mode

def extrair_contornos(img, paletaFogo = False, AreaMin = 100):
    if img is None:
        return None, None

    # Faixas de cor
    lowerFumaca = np.array([0, 0, 120])
    upperFumaca = np.array([110, 54, 255])
    lowerCentroFogo = np.array([0, 135, 115])
    upperCentroFogo = np.array([13, 255, 255])

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mascara_fumaca = cv2.inRange(img_hsv, lowerFumaca, upperFumaca)
    mascara_fogo = cv2.inRange(img_hsv, lowerCentroFogo, upperCentroFogo)
    # mascara_combinada = cv2.add(mascara_fumaca, mascara_fogo)

    # Operações morfológicas para estabilizar a máscara
    kernel = np.ones((5, 5), np.uint8)
    mascara_limpa = cv2.morphologyEx(mascara_fogo if paletaFogo else mascara_fumaca, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtro de tamanho para o treinamento
    contornos_filtrados = [c for c in contornos if cv2.contourArea(c) > AreaMin]

    if not contornos_filtrados:
        return None

    return contornos_filtrados, img_hsv

def extrair_features(img_hsv, contorno):
    mascara_pixels = np.zeros(img_hsv.shape[:2], dtype="uint8")
    cv2.drawContours(mascara_pixels, [contorno], -1, 255, -1)

    h_canal, s_canal, v_canal = cv2.split(img_hsv)

    features_textura = []
    for canal, hist_range in [(h_canal, (0, 180)), (s_canal, (0, 256)), (v_canal, (0, 256))]:
        pixels_regiao = canal[mascara_pixels == 255]
        if pixels_regiao.size == 0:
            features_textura.extend([0.0] * 5)
            continue

        hist = np.histogram(pixels_regiao, bins=hist_range[1], range=hist_range)[0]
        hist_prob = hist / (hist.sum() + 1e-15)
        entropia, media, mediana, desvio_padrao = entropy(hist_prob, base=2), np.mean(pixels_regiao), np.median(
            pixels_regiao), np.std(pixels_regiao)
        moda_val = mode(pixels_regiao, keepdims=True).mode[0]
        features_textura.extend([entropia, media, mediana, desvio_padrao, moda_val])

    return np.array(features_textura)