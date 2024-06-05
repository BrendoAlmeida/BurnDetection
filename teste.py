import os
import cv2
import numpy as np


class burnDetection:
    def __init__(self, imgDirPath, entropy):
        self.imgDirPath = imgDirPath
        self.entropy = entropy

        self.qtdFire = 0
        self.qtdNonFire = 0

        self.lowerFumaca = np.array([0, 135, 115])
        self.upperFumaca = np.array([13, 255, 255])

        self.lowerCentroFogo = np.array([20, 100, 100])
        self.upperCentroFogo = np.array([30, 255, 255])

        self.lowerCentroFogoBranco = np.array([0, 0, 200])
        self.upperCentroFogoBranco = np.array([180, 55, 255])

    def shannon_entropy(self, data):
        data = data.flatten()
        data = data[data > 0]
        data = data / data.sum()
        entropy = -np.sum(data * np.log2(data))
        return entropy

    def getContour(self):
        mascaraFumaca = cv2.inRange(self.hsvImg, self.lowerFumaca, self.upperFumaca)

        # intervalos de cor para o centro do fogo (amarelo)
        mascaraCentroFogo = cv2.inRange(self.hsvImg, self.lowerCentroFogo, self.upperCentroFogo)

        # intervalos de cor para o centro do fogo (branco)
        mascaraCentroFogoBranco = cv2.inRange(self.hsvImg, self.lowerCentroFogoBranco, self.upperCentroFogoBranco)

        # Combinar todas as máscaras
        mascara = cv2.bitwise_or(mascaraFumaca, mascaraCentroFogo)
        mascara = cv2.bitwise_or(mascara, mascaraCentroFogoBranco)

        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def contourCheck(self, contours):
        altura_frame = self.hsvImg.shape[0]
        limite_altura_ceu = int(altura_frame * 0.3)

        goodContourn = []
        for i, contour in enumerate(contours):
            # Ignorar contornos pequenos e contornos na parte superior da imagem
            # if cv2.contourArea(contour) < 500:
            #     continue

            (x, y, w, h) = cv2.boundingRect(contour)
            # if y > limite_altura_ceu:
                # Recortar a região do contorno da imagem original para calcular a entropia
                # roi = frame[y:y + h, x:x + w]
                # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ent = self.shannon_entropy(self.hsvImg[y:y + h, x:x + w])

            # Considerar um contorno como fogo se a entropia for alta o suficiente
            if ent > self.entropy:
                goodContourn.append(contour)

        return goodContourn

    def paintContours(self, contours):
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(self.img, [contour], -1, (0, 255, 255), 2)
            # # Encontrar o centro do contorno
            # M = cv2.moments(contour)
            # if M["m00"] != 0:
            #     cX = int(M["m10"] / M["m00"])
            #     cY = int(M["m01"] / M["m00"])
            # else:
            #     cX, cY = 0, 0
            #
            # # Desenhar um círculo no centro do contorno
            # cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
            # cv2.putText(frame, "Centro", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return self.img

    def start(self, show=True):
        for img in os.listdir(self.imgDirPath):
            self.img = cv2.imread(f"{self.imgDirPath}/{img}")
            self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            contour = self.contourCheck(self.getContour())

            if len(contour) == 0:
                self.qtdNonFire+=1
                continue

            self.qtdFire+=1

            if show:
                frame = self.paintContours(contour)
                cv2.imshow('Detecção de Fogo', frame)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        print(f"Chamas detectadas: {self.qtdFire}")
        print(f"Imagens sem fogo: {self.qtdNonFire}")

    def testColors(self):
        for i in range(1,260,1):
            print(i)
            self.lowerFumaca = np.array([0, 135, 115])
            self.upperFumaca = np.array([13, 255, 255])

            self.lowerCentroFogo = np.array([20, 100, 100])
            self.upperCentroFogo = np.array([30, 255, 255])

            self.lowerCentroFogoBranco = np.array([0, 0, 200])
            self.upperCentroFogoBranco = np.array([180, 55, 255])

            self.entropy = 16

            self.img = cv2.imread(f"images/queimada/1queimada4.jpg")
            self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            contour = self.contourCheck(self.getContour())
            frame = self.paintContours(contour)
            cv2.imshow('Detecção de Fogo', frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

# def shannon_entropy(img):
#     hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
#     hist = hist / hist.sum()
#     return entropy(hist, base=2)
#
#
# def detectar_fogo_fumaca(frame):
#     # Converter a imagem para o espaço de cores HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Definir intervalos de cor para a fumaça
#     lowerFumaca = np.array([0, 100, 100])
#     upperFumaca = np.array([13, 255, 255])
#     mascaraFumaca = cv2.inRange(hsv, lowerFumaca, upperFumaca)
#
#     # Definir intervalos de cor para o centro do fogo (amarelo)
#     lowerCentroFogo = np.array([20, 100, 100])
#     upperCentroFogo = np.array([30, 255, 255])
#     mascaraCentroFogo = cv2.inRange(hsv, lowerCentroFogo, upperCentroFogo)
#
#     # Definir intervalos de cor para o centro do fogo (branco)
#     lowerCentroFogoBranco = np.array([0, 0, 200])
#     upperCentroFogoBranco = np.array([180, 55, 255])
#     mascaraCentroFogoBranco = cv2.inRange(hsv, lowerCentroFogoBranco, upperCentroFogoBranco)
#
#     # Combinar todas as máscaras
#     mascara = cv2.bitwise_or(mascaraFumaca, mascaraCentroFogo)
#     mascara = cv2.bitwise_or(mascara, mascaraCentroFogoBranco)
#
#     # Encontrar contornos na máscara combinada
#     contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Filtro de altura para ignorar o céu (assumindo que o céu está na parte superior)
#     altura_frame = frame.shape[0]
#     limite_altura_ceu = int(altura_frame * 0.3)
#
#     maior_contorno = None
#     maior_area = 0
#
#     for contour in contours:
#         # Ignorar contornos pequenos e contornos na parte superior da imagem
#         # if cv2.contourArea(contour) < 500:
#         #     continue
#
#         (x, y, w, h) = cv2.boundingRect(contour)
#         if y < limite_altura_ceu:
#             continue
#
#         # Recortar a região do contorno da imagem original para calcular a entropia
#         # roi = frame[y:y + h, x:x + w]
#         # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         ent = shannon_entropy(hsv[y:y + h, x:x + w])
#
#         # Considerar um contorno como fogo se a entropia for alta o suficiente
#         if ent > 5:  # Ajuste este valor conforme necessário
#             if cv2.contourArea(contour) > maior_area:
#                 maior_area = cv2.contourArea(contour)
#                 maior_contorno = contour
#
#     # Desenhar o maior contorno que foi considerado fogo
#     if maior_contorno is not None:
#         (x, y, w, h) = cv2.boundingRect(maior_contorno)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.drawContours(frame, [maior_contorno], -1, (0, 255, 255), 2)
#
#         # Encontrar o centro do contorno
#         M = cv2.moments(maior_contorno)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#         else:
#             cX, cY = 0, 0
#
#         # Desenhar um círculo no centro do contorno
#         cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
#         cv2.putText(frame, "Centro", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return frame
#
#
# for img in os.listdir("images/queimada"):
#     # Detectar fogo na imagem
#     frame = detectar_fogo_fumaca(cv2.imread(f"images/queimada/{img}"))
#
#     # Mostrar a imagem com os contornos do fogo
#     cv2.imshow('Detecção de Fogo', frame)
#     cv2.waitKey(0)
#
#     # Sair do loop ao pressionar 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()


# contFogo = 0
# contNatural = 0
#
# for img in os.listdir("images/treinamento/FireImages"):
#     frame = detectar_fogo_fumaca(cv2.imread(f"images/treinamento/FireImages/{img}"))
#     if fireDetect(f"images/treinamento/FireImages/{img}"):
#         contFogo+=1
#     else:
#         contNatural+=1
#
# cv2.destroyAllWindows()
# print(f"Foram dedectados {contFogo} imagens com fogo e {contNatural} imagens naturais")


burnDetection = burnDetection("images/treinamento/NormalImages", 15)
burnDetection.start(False)
# burnDetection.testColors()