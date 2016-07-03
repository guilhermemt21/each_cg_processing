# -*- coding: utf-8 -*-
from __future__ import division

import os, sys, math
from lxml import etree
from PIL import Image    
from PIL import ImageDraw 

import cv2
import numpy as np
from matplotlib import pyplot as plt
from pylab import array, plot, show, axis, arange, figure, uint8 

############### Método para calcular a imagem integral, essencial para facilitar os cálculos com Haar Features.  ###############
############### Em uma imagem integral, cada pixel vale a soma dos pixels que estão acima dele e tambem à sua esquerda ###############
def integrateImage(imageWidth,imageHeight, pix, pix2) :
    pix2[0][0] = pow(pix2[0][0],2)
    for x in range(imageWidth) :
        for y in range(imageHeight) :
            if x == 0 and y != 0 :
                pix[x][y] += pix[x][y-1]
                pix2[x][y] = pix2[x][y-1] + pow(pix2[x][y],2)
            elif y == 0  and x != 0 :
                pix[x][y] += pix[x-1][y]
                pix2[x][y] = pix2[x-1][y] + pow(pix2[x][y],2)
            elif x != 0 and y != 0 :
                pix[x][y] += pix[x-1][y] + pix[x][y-1] - pix[x-1][y-1]
                pix2[x][y]=pix2[x-1][y]+pix2[x][y-1]-pix2[x-1][y-1]+pow(pix2[x][y],2)


############### Método para leitura e armazenamento das definições de Haar Features, bem como suas propriedades ###############
############### O arquivo está reorganizado de forma diferente do original, disponibilizado pelo OpenCV, apenas para melhor eficiência na leitura ##############
def parseXml() :
    stages = cascade.find("stages")
    listHaarCascade = []
    for stage in stages :
        stageList = []

        trees = stage.find("trees")
        for tree in trees :
            treeArray = []
            for idx in range(2) :
                nodeList = []
                node = tree[idx+1]
                feature = node.find("feature")
                rects = feature.find("rects")

                rectsList = ()
                for rect in rects :
                    rectTextSplit = rect.text.split()
                    rectsList += (rectTextSplit,)
                nodeList.append(rectsList)
                nodeThreshold = float(node.find("threshold").text)
                nodeList.append(nodeThreshold)
                leftValue = node.find("left_val")
                nodeList.append(leftValue)
                rightValue = node.find("right_val")
                nodeList.append(rightValue)
                leftNode = node.find("left_node")
                nodeList.append(leftNode)
                rightNode = node.find("right_node")
                nodeList.append(rightNode)

                treeArray.append(nodeList)

            stageList.append(treeArray)

        stageThreshold = float(stage.find("stage_threshold").text)
        stageList.append(stageThreshold)
        listHaarCascade.append(stageList)

    return listHaarCascade


############### Método responsável pela validação de uma região com Haar Features ###############
############### Aqui a imagem integral mostra sua eficiência, possibilitando que o cálculo com a convolução da Feature seja sempre constante  ###############
def evalFeature(windowX,windowY,windowWidth,windowHeight,rects,scale, pix, pix2) :
    invArea=1/(windowWidth*windowHeight)

    featureSum = 0
    #print "windowX: {}, windowWidth: {},  windowY: {}, windowHeight: {}".format(windowX, windowWidth, windowY, windowHeight)
    totalX=pix[windowX+windowWidth][windowY+windowHeight] +pix[windowX][windowY] -pix[windowX+windowWidth][windowY] -pix[windowX][windowY+windowHeight]
    totalX2=pix2[windowX+windowWidth][windowY+windowHeight] +pix2[windowX][windowY] -pix2[windowX+windowWidth][windowY] -pix2[windowX][windowY+windowHeight]

    vnorm=totalX2*invArea-pow(totalX*invArea,2)
    if vnorm > 1: vnorm = math.sqrt(vnorm)
    else        : vnorm = 1
    for rect in rects:
        x = int(scale*int(rect[0]))
        y = int(scale*int(rect[1]))
        width = int(scale*int(rect[2]))
        height = int(scale*int(rect[3]))
        weight = float(rect[4])
        featureSum += weight * (pix[windowX+x+width][windowY+y+height] + pix[windowX+x][windowY+y] - pix[windowX+x+width][windowY+y] - pix[windowX+x][windowY+y+height])
    return featureSum,invArea,vnorm


############### Cada estágio possui diversas validações 'fracas'. Para passar por um estágio, é necessário que a região seja aprovada em todas as Features ###############
############### Tanto a validação individual com cada Haar Feature, quanto a validação de uma etapa inteira, possuem um limite de erro informado  ###############
############### Passando em todas as etapas definidas no arquivo xml do OpenCV, consideramos que a região é uma face ###############
def evalStages(windowX,windowY,windowWidth,windowHeight,scale, pix, pix2) :
    stagePass = True
    for stage in listHaarCascade:
        stageThreshold = stage[-1]
        stageSum = 0

        for tree in stage[:-1]:
            treeValue = 0
            idx = 0

            while True:
                node = tree[idx]
                rects = node[0]
                nodeThreshold = node[1]
                leftValue = node[2]
                rightValue = node[3]
                leftNode = node[4]
                rightNode = node[5]

                featureSum,invArea,vnorm = evalFeature(windowX,windowY, windowWidth, windowHeight,rects,scale, pix, pix2)

                if featureSum*invArea < nodeThreshold*vnorm:
                    if leftNode is None:
                        treeValue = float(leftValue.text)
                        break
                    else:
                        idx = int(leftNode.text)
                else:
                    if rightNode is None:
                        treeValue = float(rightValue.text)
                        break
                    else:
                        idx = int(rightNode.text)

            stageSum += treeValue
        stagePass = stageSum >= stageThreshold

        if not stagePass :
            return stagePass

    return stagePass



############### O algoritmo principal para identificar uma face, possui 3 loops principals ###############
############### O primeiro loop, é responsável por aumentar o tamanho da janela em que trabalhamos, multiplicando-a por 2,4  ###############
############### O segundo loop e terceiro loops são responsáveis pelo movimento da janela, invocando o método de reconhecimento a cada passo ###############
def detect(imageWidth,imageHeight, pix, pix2) :
    listResult = []
    scale, scaleFactor = 1, 1.25
    windowWidth, windowHeight = (int(n) for n in cascade.find("size").text.split())

    while windowWidth < imageWidth and windowHeight < imageHeight:
        windowWidth = windowHeight = int(scale*20)
        step = int(scale*2.4)
        windowX = 0
        while windowX < imageWidth-scale*24:
            windowY = 0
            while windowY < imageHeight-scale*24:
                if evalStages(windowX,windowY,windowWidth,windowHeight,scale, pix, pix2):
                    listResult.append((windowX, windowY, windowWidth, windowHeight))
                windowY += step
            windowX += step
        scale = scale * scaleFactor
    return listResult



############### Método responsável por reduzir múltiplas detecções sobre a mesma área em uma única ocorrência ###############
def simplifyRects(listResult):
    return listResult
    simplifiedList=[]
    if len(listResult)>0:
        centerList = []
        maxX,maxY,maxWidth,maxHeight=listResult[0]
        centerList.append((maxX+maxWidth/2,maxY+maxHeight/2))
        simplifiedList.append((listResult[0]))
        for rect in listResult :
            x,y,width,height = rect
            for center in centerList :
                centerX, centerY = center
                if x < centerX < x+width and y < centerY < y + height :
                    break
                elif x+width<maxX or maxX+maxWidth<x or y+height<maxY or maxY+maxHeight<y:
                    maxX=x
                    maxY=y
                    maxWidth=width
                    maxHeight=height
                    simplifiedList.append((rect))
                    centerList.append((maxX+maxWidth/2,maxY+maxHeight/2))
        return simplifiedList
    else:
        return []

############### Finalmente, contorna a região detectada ###############
def drawRect(simplifiedList, im) :
    lineWidth = 1
    rects = 0
    for rect in simplifiedList :
        rects += 1
        windowX,windowY,windowWidth,windowHeight = rect
        draw = ImageDraw.Draw(im)
        draw.line((windowX, windowY, windowX, windowY+windowHeight),fill = 128, width = lineWidth)
        draw.line((windowX, windowY+lineWidth/2, windowX+windowWidth, windowY+lineWidth/2),fill = 128, width = lineWidth)
        draw.line((windowX+windowWidth,windowY,windowX+windowWidth,windowY+windowHeight),fill = 128, width = lineWidth)
        draw.line((windowX, windowY+windowHeight-lineWidth/2, windowX+windowWidth, windowY+windowHeight-lineWidth/2),fill = 128, width = lineWidth)
        del draw
    print "{} faces detectadas!".format(rects)
    print "-----------------------"


############### Algoritmo para invocar métodos principais dos procedimentos do Viola Jones ###############
############### Os métodos aqui invocados são uma adapatação do código original, veja em: https://github.com/FlorentRevest/ViolaJones  ###############
def violaJonesDetection(im, isRGB) :
    imageWidth, imageHeight = im.size
    pixels = im.load()

    grayLevelR = 30
    grayLevelG = 59
    grayLevelB = 11 
    
    #Algumas imagens chegam em formato RGB, outras em single channel. pix e pix2 são as matrizes de pixels para processamento posterior
    print "width:{} , height:{}".format(imageWidth, imageHeight)
    pix = [[0 for y in range(imageHeight)] for x in range(imageWidth)] 
    pix2 = [[0 for y in range(imageHeight)] for x in range(imageWidth)] 
    if isRGB :
        for y in range(imageHeight):
            for x in range(imageWidth):
                pix[x][y] = int((grayLevelR*pixels[x,y][0]+grayLevelG*pixels[x,y][1]+grayLevelB*pixels[x,y][2])/100)
                pix2[x][y] = pix[x][y]
    else :
        for y in range(imageHeight):
            for x in range(imageWidth):
                pix[x][y] = pixels[x,y]
                pix2[x][y] = pix[x][y]
    
    integrateImage(imageWidth,imageHeight, pix, pix2)

    listResult = detect(imageWidth,imageHeight, pix, pix2)
    chosenList = simplifyRects(listResult)
    drawRect(chosenList, im)

    im.show()



############### Main ###############
imgPath = raw_input('* Insira o nome da imagem, incluindo sua extensão: ') 
brightness = float(input('* Insira o fator de brilho [0.1-3.0]: ')) 
blur = int(input('* Insira o fator de blur [1-30]: ')) 
kernelSize = int(input('* Insira o tamanho do kernel (apenas números Impares!): '))
quantizationK = int(input('* Insira o K para quantização [1- 255]: '))
originalImage = Image.open(imgPath)




############### Cria imagens de com diferentes modos de blur com o fator informado###############
blurOutputString = 'blur.jpg'
medianBlurOutputString = 'medianBlur.jpg'
bilateralBlurOutputString = 'bilateralBlur.jpg'
img = cv2.imread(imgPath)

blurOutput = cv2.blur(img,(blur,blur))
medianOutput = cv2.medianBlur(img, (blur if blur % 2 == 1 else blur+1))
bilateralOutput = cv2.bilateralFilter(img,blur,blur*2, blur/2)

cv2.imwrite(blurOutputString, blurOutput)
cv2.imwrite(medianBlurOutputString, medianOutput)
cv2.imwrite(bilateralBlurOutputString, bilateralOutput)




############### Cria imagem com o fator de brilho informado ###############
brightnessOutputString = 'brightness.jpg'
brightnessImage = cv2.imread(imgPath)
x = arange(255.0) 

brightnessOutput = 255.0*(brightnessImage/255)**brightness
brightnessOutput = array(brightnessOutput,dtype=uint8)

cv2.imwrite(brightnessOutputString,brightnessOutput)




 
############### Cria imagem com o fator de quantização informado ###############
quantizationOutputString = 'quantization.jpg'
quantizationImage = cv2.imread(imgPath)
Z = quantizationImage.reshape((-1,3))
Z = np.float32(Z)
 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(Z,quantizationK,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
 
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((quantizationImage.shape))

cv2.imwrite(quantizationOutputString,res2)

############### Cria imagem com o fator de quantização e brilho ao mesmo tempo ###############
quantizationBrightnessOutputString = 'quantization_brightness.jpg'
quantizationBrightness = cv2.imread(quantizationOutputString)
Z = quantizationBrightness.reshape((-1,3))
Z = np.float32(Z)
 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(Z,quantizationK,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
 
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((quantizationBrightness.shape))

cv2.imwrite(quantizationBrightnessOutputString,res2)



############### Cria imagens com aplicação dos filtros laplaciano e sobel (x e y) ###############
grayscaleOutputString = 'grayscale.jpg'
laplacianOutputString = 'laplacian.jpg'
sobelXOutputString = 'sobelX.jpg'
grayImage = cv2.cvtColor(np.array(originalImage), cv2.COLOR_BGR2GRAY)

#remove ruído
img = cv2.GaussianBlur(grayImage,(3,3),0)

#aplicação de kernels
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=kernelSize)  # x
#sobely = cv2.Sobel(img,cv2.CV_16S,0,1,ksize=kernelSize)  # y

cv2.imwrite(grayscaleOutputString,img)
cv2.imwrite(laplacianOutputString,laplacian)
cv2.imwrite(sobelXOutputString,sobelx)




############### Invoca o algoritmo de viola jones para todas as imagens criadas ###############
#carrega arquivo com definições das haar features
print "Inicializando Viola Jones Algorithm"
cascade = etree.parse("haarcascade_frontalface_custom.xml").getroot().find("haarcascade_frontalface_custom")
listHaarCascade = parseXml()

print "Processando imagem: ORIGINAL"
violaJonesDetection(originalImage, True)

print "Processando imagem: fator de BLUR"
violaJonesDetection(Image.open(blurOutputString), True)

print "Processando imagem: fator de BLUR (MEDIANA)"
violaJonesDetection(Image.open(medianBlurOutputString), True)

print "Processando imagem: fator de BLUR (BILATERAL)"
violaJonesDetection(Image.open(bilateralBlurOutputString), True)

print "Processando imagem: fator de BRILHO"
violaJonesDetection(Image.open(brightnessOutputString), True)

print "Processando imagem: fator de QUANTIZAÇÂO"
violaJonesDetection(Image.open(quantizationOutputString), True)

print "Processando imagem: fator de QUANTIZAÇÂO e BRILHO"
violaJonesDetection(Image.open(quantizationBrightnessOutputString), True)

print "Processando imagem: GRAYSCALE"
violaJonesDetection(Image.open(grayscaleOutputString), False)

print "Processando imagem: filtro LAPLACIANO"
violaJonesDetection(Image.open(laplacianOutputString), False)

print "Processando imagem: filtro SOBEL em X"
violaJonesDetection(Image.open(sobelXOutputString), False)

