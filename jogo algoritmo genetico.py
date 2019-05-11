import time
import numpy as np
import pygame
from sklearn.neural_network import MLPClassifier

#em caracteristicas, o primeiro valor é x, o segundo é y, o terceiro é x2,
#o quarto é y2
caracteristicas= []
result_esperado= []

#Classificações de saída da rede neural: -1= esquerda, 0= direita, 1= cima, 2= baixo

x= 50
y= 50

x2= 500
y2= 500

altura= 40
largura= 60

vel= 10
vel2= 5

pygame.init()

janela= pygame.display.set_mode((800, 600))
pygame.display.set_caption('teste I.A.')

#criação de blocos np canvas: x, y, x2, y2 são posições iniciais
bloco1= pygame.draw.rect(janela, (0, 0, 255), (x, y, largura, altura))
bloco2= pygame.draw.rect(janela, (255, 0, 0), (x2, y2, largura, altura))

pygame.font.init()

fonte_padrao= pygame.font.get_default_font()
fonte= pygame.font.SysFont(fonte_padrao, 30)

for i in range(0, 1500):
    pygame.time.delay(20)
    
    for event in pygame.event.get():
        if event.type== pygame.QUIT:
            run= False

    botao= pygame.key.get_pressed()

    janela.fill((0, 0, 0))

    texto= fonte.render('Coletando dados... '+str(i)+' dado(s) coletado(s)', 1, (255, 255, 255))
    janela.blit(texto, (450, 0))

    pygame.draw.rect(janela, (0, 0, 255), (x, y, largura, altura))
    pygame.draw.rect(janela, (255, 0, 0), (x2, y2, largura, altura))
    
    pygame.display.update()
    botao_atual= 3

    if botao[pygame.K_LEFT] and x!= 0:
        x-= vel
        botao_atual= -1
        print('esquerda')
    elif botao[pygame.K_RIGHT] and (x+ largura)!= 800:
        x+= vel
        botao_atual= 0
        print('direita')
    elif botao[pygame.K_DOWN] and (y+ altura)!= 600:
        y+= vel
        botao_atual= 2
        print('baixo')
    elif botao[pygame.K_UP] and y!= 0:
        y-= vel
        botao_atual= 1
        print('cima')

    if y2> y:
        y2-= vel2
    if y2< y:
        y2+= vel2
    if x2> x:
        x2-= vel2
    if x2< x:
        x2+= vel2

    caracteristicas.append([float(x), float(y), float(x2), float(y2), float(y2- y), float(x2- x), float(800- x), float(600- y)])
    result_esperado.append([botao_atual])

caracteristicas= np.ravel(caracteristicas)
result_esperado= np.ravel(result_esperado)
caracteristicas= np.reshape(caracteristicas, (1500, 8))

print(caracteristicas)
print(result_esperado)

texto= fonte.render('Dados coletados', 1, (255, 255, 255))
janela.blit(texto, (300, 300))
pygame.display.update()
pygame.time.delay(2000)

for i in range(1, 800):
    
    classificador= MLPClassifier(solver= 'adam', hidden_layer_sizes= (5,),
                                     max_iter= i)

    classificador.fit(caracteristicas, result_esperado)

    x= 50
    y= 50

    x2= 500
    y2= 500
    
    for j in range(0, 400):

        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                run= False

        janela.fill((0, 0, 0))
        
        pygame.draw.rect(janela, (0, 0, 255), (x, y, largura, altura))
        pygame.draw.rect(janela, (255, 0, 0), (x2, y2, largura, altura))

        texto= fonte.render('Iteração: '+str(i), 1, (255, 255, 255))
        janela.blit(texto, (0, 0))
        
        pygame.display.update()

        resultado= classificador.predict([[float(x), float(y), float(x2), float(y2), float(y2- y), float(x2- x), float(800- x), float(600- y)]])
        
        if resultado[0]== -1 and x!= 0:
            x-= vel
        elif resultado[0]== 0 and (x+ largura)!= 800:
            x+= vel
        elif resultado[0]== 1 and y!= 0:
            y-= vel
        elif resultado[0]== 2 and (y+ altura)!= 600:
            y+= vel
        
        if y2> y:
            y2-= vel2
        if y2< y:
            y2+= vel2
        if x2> x:
            x2-= vel2
        if x2< x:
            x2+= vel2

pygame.quit()
    
