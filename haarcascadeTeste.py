import cv2

img = cv2.imread("images/queimada/queimada4.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

classificador = cv2.CascadeClassifier("classificador/cascade.xml")

fogo = classificador.detectMultiScale(img, 12, 5)

print(len(fogo))