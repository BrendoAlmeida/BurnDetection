import os
import time

import cv2
import numpy as np


class burnDetection:
    def __init__(self, imgDirPath, entropy):
        self.imgDirPath = imgDirPath
        self.entropy = entropy

        self.qtdFire = 0
        self.qtdNonFire = 0

        # self.lowerFumaca = np.array([0, 135, 115])
        # self.upperFumaca = np.array([13, 255, 255])
        #
        # self.lowerCentroFogo = np.array([20, 100, 100])
        # self.upperCentroFogo = np.array([30, 255, 255])
        #
        # self.lowerCentroFogoBranco = np.array([0, 0, 200])
        # self.upperCentroFogoBranco = np.array([180, 55, 255])

        self.lowerFumaca = np.array([0, 0, 153])
        self.upperFumaca = np.array([180, 55, 244])

        self.lowerCentroFogo = np.array([0, 135, 115])
        self.upperCentroFogo = np.array([13, 255, 255])

        # self.lowerCentroFogoBranco = np.array([0, 0, 200])
        # self.upperCentroFogoBranco = np.array([180, 55, 255])

    def shannon_entropy(self, data):
        data = data.flatten()
        data = data[data > 0]
        data = data / data.sum()
        entropy = -np.sum(data * np.log2(data))
        return entropy

    def getContour(self, fumaca=False):
        # intervalos de cor para a fumaça
        mascara = cv2.inRange(self.hsvImg, self.lowerFumaca, self.upperFumaca)

        if fumaca:
            # intervalos de cor para o centro do fogo (amarelo)
            mascaraFogo = cv2.inRange(self.hsvImg, self.lowerCentroFogo, self.upperCentroFogo)

            # intervalos de cor para o centro do fogo (branco)
            # mascaraCentroFogoBranco = cv2.inRange(self.hsvImg, self.lowerCentroFogoBranco, self.upperCentroFogoBranco)

            # Combinar todas as máscaras
            # mascara = cv2.bitwise_or(mascaraFogo, mascaraCentroFogoBranco)
            mascara = mascaraFogo

        contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def contourCheck(self, contours):
        goodContourn = []
        for i, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
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
        for imgName in os.listdir(self.imgDirPath):
            # Teste para fumaça
            self.img = cv2.imread(f"{self.imgDirPath}/{imgName}")
            self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            contour = self.contourCheck(self.getContour())

            if len(contour) == 0:
                self.qtdNonFire+=1
                continue

            # Em caso de fumaça verificar fogo
            self.qtdFire+=1
            cv2.putText(self.img, 'Fumaca!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            contourFogo = self.contourCheck(self.getContour(True))

            if len(contourFogo) > 0:
                cv2.putText(self.img, 'Fogo!', (self.img.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if show:
                frame = self.paintContours(contour)
                cv2.imshow('Detecção de Fogo', frame)
                cv2.waitKey(0)

                frame = self.paintContours(contourFogo)
                cv2.imshow('Detecção de Fogo', frame)
                cv2.waitKey(0)

        cv2.destroyAllWindows()
        print(f"Chamas detectadas: {self.qtdFire}")
        print(f"Imagens sem fogo: {self.qtdNonFire}")

    def startVideo(self, show=True):
        for video in os.listdir(self.imgDirPath):
            self.video = cv2.VideoCapture(f"{self.imgDirPath}/{video}")

            while self.video.isOpened():
                ret, self.img = self.video.read()
                if not ret:
                    break

                self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
                contour = self.contourCheck(self.getContour())

                if len(contour) == 0:
                    self.qtdNonFire+=1
                    continue

                self.qtdFire += 1
                cv2.putText(self.img, 'Fumaca!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                contourFogo = self.contourCheck(self.getContour(True))

                if len(contourFogo) > 0:
                    cv2.putText(self.img, 'Fogo!', (self.img.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2, cv2.LINE_AA)

                if show:
                    cv2.imshow('Detecção de Fumaça', self.img)
                    frame = self.paintContours(contour)
                    cv2.waitKey(0)

                    frame = self.paintContours(contourFogo)
                    cv2.imshow('Detecção de Fogo', frame)
                    cv2.waitKey(0)

            self.video.release()

        cv2.destroyAllWindows()
        print(f"Chamas detectadas: {self.qtdFire}")
        print(f"Imagens sem fogo: {self.qtdNonFire}")

    def testColors(self):
        for i in range(0,260,1):
            print(i)
            self.lowerFumaca = np.array([0, 0, 153])
            self.upperFumaca = np.array([180, 55, 244])

            self.lowerCentroFogo = np.array([20, 100, 100])
            self.upperCentroFogo = np.array([30, 255, 255])

            self.lowerCentroFogoBranco = np.array([0, 0, 200])
            self.upperCentroFogoBranco = np.array([180, 55, 255])

            self.entropy = 16

            self.img = cv2.imread(f"images/frameVideo/Fire/fumaca2.png")
            self.hsvImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            contour = self.contourCheck(self.getContour())
            frame = self.paintContours(contour)
            cv2.imshow('Detecção de Fogo', frame)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


# burnDetection = burnDetection("images/treinamento/FireImages", 16)
# burnDetection.start(False)
# burnDetection.testColors()

burnDetection = burnDetection("video/Fire", 16)
burnDetection.startVideo(False)