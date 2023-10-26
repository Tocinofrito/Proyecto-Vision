import cv2
#video captura de la deteccion
captura = cv2.videoCapture(0)
captura.set(3,1280)
captura.set(4,720)

#ciclo infinito
while true:
    #realiza la lectura de la videocaptura

    ret, frame = captura.read()
    if ret == false:
        break

    #extraer el ancho y alto de los fotogramas
    al, an, c = frame.shape

    #tomando el centro de la imagen
    #en x
    x1 = int(an / 3) #tomamos 1/3 de la imagen
    x2 = int(x1 * 2) #hasta el inicio del 3/3 de la imagen

    #en y:
    y1 = int(al / 3) #tomamos 1/3 de la imagen
    y2 = int(y1 * 2) #hasta el inicio del 3/3 de la imagen

    #texto
    cv2.putText(frame, 'Ubique el ojo en el rectangulo,', (x1 - 50, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)

    #ubicar el rectangulo en las zonas extraidas
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)

    #realizamos un recorte a nuestra zona de interes
    recorte = frame(y1:y2, x1:x2)

    #pasamos el recorte a escala de grises
    gris = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    #aplicamos un filtro Gaussiano para eliminar las pestañas
    gris = cv2.GaussianBlur(gris, (3,3), 0) #entre mayor sea el kernel mas se desenfoca

    #aplicaremos un umbral para detectar la pupila por el color
    _, umbral = cv2.thereshold(gris, 7, 255, cv2.THRESH_BINARY_INV)

    #extraemos los contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contornos)

    #extaer area de los contornos
    #ordenar del mas grande al mas pequeño
    contornos = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    #ibujamos los contornos extraidos
    for contorno in contornos:
        #dibujamos el contorno
        #cv2.drawContours(recorte, [contorno], -1, (0,255,0),2)

        #dibujamos el rectangulo a partir del contorno
        #extraemos las coordenadas
        (x, y, ancho, alto) = cv2.boundinRect(contorno)
        #dibujamos
        cv2.rectangle(frame, (x +x1, y + y1), (x + ancho + x1, y + alto + y1), (0,255,0),1)

        #mostramos dos lineas a partir del centro del ojo
        #eje y:
        cv2.line(frame, (x1 + x + int(ancho/2),0), (x1 + x + int(ancho/2), al), (0,0,255),1)
        #eje x:
        cv2.line(frame, (y1 + y + int(ancho/2),0), (an, y1 + y + + int(ancho/2)), (0,0,255),1)
        break

        #mostramos el recorte en gris
        cv2.imshow("ojos", frame)

        #mostramos el recorte 
        cv2.imshow("Recorte", recorte)

        #mostramos el umbral
        cv2.imshow("Umbral", umbral)

        t = cv2.waitKey(1)

        if t == 27:
            break

    captura.release()
    cv2.destroyAllWindows()