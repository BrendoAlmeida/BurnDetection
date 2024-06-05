# Importar bibliotecas
import cv2
import numpy as np
import os

def shannon_entropy(data):
    data = data.flatten()
    data = data[data > 0]
    data = data / data.sum()
    entropy = -np.sum(data * np.log2(data))
    return entropy


def fireDetect(imgPath, video=False, entopyVal = 14):
    area = 0
    cont_frame = 0
    imagem = cv2.imread(imgPath)
    aux_area = 0

    # Transformar imagem para escala HSV
    imgHSV = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    lowerFumaca = np.array([0, 100, 100])
    upperFumaca = np.array([13, 255, 255])
    mascaraFumaca = cv2.inRange(imgHSV, lowerFumaca, upperFumaca)

    contornos, hierarchy = cv2.findContours(mascaraFumaca, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar variável para desenho na imagem
    desenho = np.zeros_like(imagem)

    # Verificar se existem contornos na imagem
    if len(contornos) == 0:
        print("No frame", cont_frame, ", nenhum fogo foi detectado.") # Caso não existam contornos, nenhum fogo estará sendo mostrado
        return
    else:
        maior_contorno = -1
        for i, contorno in enumerate(contornos):
            area = cv2.contourArea(contorno)

            x, y, w, h = cv2.boundingRect(contorno)
            entropy = shannon_entropy(imgHSV[y:y + h, x:x + w])

            # Comparar área do i-ésimo contorno com a área de referência, incialmente 0
            if area >= aux_area and entropy > entopyVal:
                aux_area = area
                maior_contorno = i

        if maior_contorno == -1:
            return

        color = (0, 0, 255)
        contornos_conectados = [contornos[maior_contorno]]
        for posContorno, contorno in enumerate(contornos):
            # if cv2.contourArea(contorno) > 50: # remove um pouco do ruido
            centro1 = tuple(contorno[:, 0].mean(axis=0))
            centro2 = tuple(contornos[maior_contorno][:, 0].mean(axis=0))
            distancia = np.linalg.norm(np.array(centro1) - np.array(centro2))

            if distancia < 100:
                contornos_conectados.append(contorno)

        cv2.drawContours(desenho, contornos_conectados, -1, color, 1, 8, hierarchy, 0)
        cv2.drawContours(imagem, contornos_conectados, -1, color, 1, 8, hierarchy, 0)

        # if entropy < 14:
        #     return entropy

        if aux_area > 0:
            print(f"Fogo detectado no frame {cont_frame}, o maior fogo possui uma área igual a {aux_area}")
            return True
        else:
            print("No frame", cont_frame, ", nenhum fogo foi detectado.")
            return False

    cv2.imshow("Video", mascaraFumaca)
    cv2.imshow("Teste", desenho)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Video2", imagem)
    cv2.waitKey(0)
    cont_frame = cont_frame + 1

    cv2.destroyAllWindows()


# fireDetect("images/queimada/1queimada4.jpg")
# fireDetect("images/queimada/PublicDataset01329.jpg")

# for i in range(10, 255, 10):
#     print(i)
#     hmin, smin, vmin = 0, 0, 100
#     hmax, smax, vmax = 180, 55, 255
#     fireDetect("images/queimada/queimada4.jpg", entopyVal=0)
    # fireDetect("images/queimada/PublicDataset01329.jpg", entopyVal=0)


contFogo = 0
contNatural = 0

for img in os.listdir("images/treinamento/NormalImages"):
    # entropy = fireDetect(f"images/treinamento/NormalImages/{img}")
    if fireDetect(f"images/treinamento/NormalImages/{img}"):
        contFogo+=1
    else:
        contNatural+=1
    print(f"Foram dedectados {contFogo} imagens com fogo e {contNatural} imagens naturais")