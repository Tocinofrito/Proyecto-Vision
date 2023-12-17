import cv2
import numpy as np

captura = cv2.VideoCapture(0)
captura.set(3, 1280)
captura.set(4, 720)

# Load your custom image
custom_image = cv2.imread('itachi.png', cv2.IMREAD_UNCHANGED)
custom_image = cv2.resize(custom_image, (50, 50))  # Resize as needed

while True:
    ret, frame = captura.read()
    if not ret:
        break

    al, an, c = frame.shape

    x1 = int(an / 3)
    x2 = int(x1 * 2)

    y1 = int(al / 3)
    y2 = int(y1 * 2)

    cv2.putText(frame, 'Ubique el ojo en el rectangulo,', (x1 - 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    recorte = frame[y1:y2, x1:x2]

    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(gris, 30, 255, cv2.THRESH_BINARY)
    umbral = cv2.GaussianBlur(umbral, (5, 5), 0)

    contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contorno in contornos:
        (x, y, ancho, alto) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x + x1, y + y1), (x + ancho + x1, y + alto + y1), (0, 255, 0), 1)

        circulos = cv2.HoughCircles(gris[y:y + alto, x:x + ancho], cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

        if circulos is not None:
            circulos = np.uint16(np.around(circulos))

            for i in circulos[0, :]:
                cv2.circle(frame, (i[0] + x + x1, i[1] + y + y1), i[2], (0, 255, 0), 2)

                # Asegurarse de que las coordenadas estén dentro del rango
                x_pupil = max(0, min(i[0] - i[2] + x1, frame.shape[1]))
                y_pupil = max(0, min(i[1] - i[2] + y1, frame.shape[0]))

                # Redimensionar la imagen personalizada al tamaño del círculo
                custom_image_resized = cv2.resize(custom_image, (2 * i[2], 2 * i[2]))

                # Extraer la región de la pupila
                pupil_region = frame[y_pupil:y_pupil + 2 * i[2], x_pupil:x_pupil + 2 * i[2]]

                # Verificar si las dimensiones coinciden y realizar la asignación
                if pupil_region.shape == custom_image_resized.shape:
                    frame[y_pupil:y_pupil + 2 * i[2], x_pupil:x_pupil + 2 * i[2]] = pupil_region * (1 - custom_image_resized[:, :, 3] / 255.0) + custom_image_resized[:, :, :3] * (custom_image_resized[:, :, 3] / 255.0)

    cv2.imshow("ojos", frame)
    cv2.imshow("Umbral", umbral)

    t = cv2.waitKey(1)

    if t == 27:
        break

captura.release()
cv2.destroyAllWindows()
