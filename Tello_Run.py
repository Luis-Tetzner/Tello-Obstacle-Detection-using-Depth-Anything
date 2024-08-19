import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Função para aplicar limiarização a uma imagem em escala de cinza com um limiar parametrizável
def aplicar_limiarizacao(imagem, limiar=18.0):
    imagem_limiarizada = np.zeros_like(imagem, dtype='uint8')
    imagem_limiarizada[imagem > limiar] = 255
    return imagem_limiarizada

# Função para desenhar um retângulo na imagem
def desenhar_retangulo(imagem, coords_retangulo, sem_obstaculos):
    cor = (0, 255, 0) if sem_obstaculos else (0, 0, 255)
    espessura = 2
    cv2.rectangle(imagem, coords_retangulo[0], coords_retangulo[1], cor, espessura)

# Função para determinar o melhor caminho com base nas coordenadas dos sub-retângulos
def determinar_melhor_caminho(coords_sub_retangulo):
    direcoes = []
    if coords_sub_retangulo == (0, 0):
        direcoes.append("esquerda e para cima")
    if coords_sub_retangulo == (0, 1):
        direcoes.append("esquerda e para baixo")
    if coords_sub_retangulo == (1, 0):
        direcoes.append("direita e para cima")
    if coords_sub_retangulo == (1, 1):
        direcoes.append("direita e para baixo")
    return direcoes

# Carregar e configurar o modelo Depth Anything
encoder = 'vits'
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
depth_anything = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(dispositivo)
depth_anything.eval()

# Transformações para preparar a imagem para o modelo
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Tamanho do retângulo principal
largura_retangulo = 200
altura_retangulo = 200

# Tamanhos dos sub-retângulos
largura_sub_retangulo = largura_retangulo // 2
altura_sub_retangulo = altura_retangulo // 2

while cap.isOpened():
    ret, imagem_bruta = cap.read()
   
    if not ret:
        break

    imagem_bruta = cv2.resize(imagem_bruta, (640, 480))
    imagem = cv2.cvtColor(imagem_bruta, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = imagem.shape[:2]
    
    imagem = transform({'image': imagem})['image']
    imagem = torch.from_numpy(imagem).unsqueeze(0).to(dispositivo)
    
    with torch.no_grad():
        profundidade = depth_anything(imagem)
    
    profundidade = F.interpolate(profundidade[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    profundidade_clonada = torch.clone(profundidade).cpu().numpy()
    profundidade = (profundidade - profundidade.min()) / (profundidade.max() - profundidade.min()) * 255.0
    
    profundidade = profundidade.cpu().numpy().astype(np.uint8)
    profundidade_colorida = cv2.applyColorMap(profundidade, cv2.COLORMAP_INFERNO)

    # Aplicar limiarização na imagem de profundidade
    profundidade_limiarizada = aplicar_limiarizacao(profundidade_clonada)
    
    # Coordenadas do retângulo principal centralizado
    x_retangulo = int((w - largura_retangulo) / 2)
    y_retangulo = int((h - altura_retangulo) / 2)
    coords_retangulo = ((x_retangulo, y_retangulo), (x_retangulo + largura_retangulo, y_retangulo + altura_retangulo))
    
    # Verificar se o caminho está livre de obstáculos
    sem_obstaculos = np.mean(profundidade_limiarizada[y_retangulo:y_retangulo + altura_retangulo, x_retangulo:x_retangulo + largura_retangulo]) < 128
    
    # Desenhar o retângulo principal na imagem de profundidade
    desenhar_retangulo(profundidade_colorida, coords_retangulo, sem_obstaculos)

    # Desenhar o retângulo principal na imagem original
    desenhar_retangulo(imagem_bruta, coords_retangulo, sem_obstaculos)

    # Se houver obstáculos, dividir o retângulo principal em quatro metades e verificar qual metade está livre
    if not sem_obstaculos:
        sub_retangulos_livres = []  # Lista para armazenar os sub-retângulos livres

        for i in range(2):
            for j in range(2):
                # Coordenadas dos retângulos secundários
                x_sub_retangulo = x_retangulo + i * largura_sub_retangulo
                y_sub_retangulo = y_retangulo + j * altura_sub_retangulo
                coords_sub_retangulo = (i, j)

                # Calcular a média da intensidade dos pixels para cada sub-retângulo
                media_intensidade = np.mean(profundidade_limiarizada[y_sub_retangulo:y_sub_retangulo + altura_sub_retangulo, x_sub_retangulo:x_sub_retangulo + largura_sub_retangulo])

                # Verificar se o sub-retângulo está livre de obstáculos
                sub_retangulo_livre = media_intensidade < 128

                # Se o sub-retângulo estiver livre, adicionar às listas correspondentes
                if sub_retangulo_livre:
                    sub_retangulos_livres.append((media_intensidade, coords_sub_retangulo))

                # Desenhar o sub-retângulo na imagem de profundidade
                desenhar_retangulo(profundidade_colorida, ((x_sub_retangulo, y_sub_retangulo), (x_sub_retangulo + largura_sub_retangulo, y_sub_retangulo + altura_sub_retangulo)), sub_retangulo_livre)

                # Desenhar o sub-retângulo na imagem original
                desenhar_retangulo(imagem_bruta, ((x_sub_retangulo, y_sub_retangulo), (x_sub_retangulo + largura_sub_retangulo, y_sub_retangulo + altura_sub_retangulo)), sub_retangulo_livre)

        # Se houver pelo menos dois sub-retângulos livres
        if len(sub_retangulos_livres) >= 2:
            # Ordenar os sub-retângulos livres com base na média de intensidade (do mais escuro para o mais claro)
            sub_retangulos_livres.sort(key=lambda x: x[0])

            # Escolher o sub-retângulo mais escuro como a direção preferida
            melhor_sub_retangulo = sub_retangulos_livres[0][1]

            # Determinar o melhor caminho com base no sub-retângulo mais escuro
            melhor_caminho = determinar_melhor_caminho(melhor_sub_retangulo)
            print("Melhor caminho: ", melhor_caminho)

    # Concatenar as duas imagens horizontalmente
    imagem_combinada = np.concatenate((imagem_bruta, profundidade_colorida), axis=1)

    # Exibir as duas imagens em uma janela
    cv2.imshow('Imagens Combinadas', imagem_combinada)

    # Pressione 'q' para sair
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()