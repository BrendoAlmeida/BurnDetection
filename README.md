# 🔥 BurnIA - Detecção de Fogo e Fumaça em Vídeos

Projeto voltado para a extração de características visuais (features) de vídeos contendo fogo ou cenas normais, com o objetivo de treinar modelos de aprendizado de máquina para detecção automática de incêndios.

## 📁 Estrutura do Projeto

```
BurnIA/
├── Datasets/                 # Local onde serão salvos os datasets processados
├── Frames2Treinamento/      # Frames extraídos dos vídeos para uso no treinamento
├── FrameVideos/             # Pode conter os frames separados por vídeo
├── Modelos/
│   └── modelo_features     # Armazenamento de modelos treinados
├── Videos/
│   ├── Fire/
│   │   ├── Fire1.mp4
│   │   └── Fire2.mp4        # Vídeos contendo cenas com fogo
│   └── Normal/
│       ├── Normal1.mp4
│       └── Normal2.mp4      # Vídeos com cenas normais, sem fogo
├── dataset.ipynb            # Notebook para carregar e explorar os dados do dataset
├── extratorFeatures.py      # Script para extração de contornos e features das regiões de interesse
├── main.ipynb               # Pipeline principal para processamento dos vídeos
├── outros.ipynb             # Notas ou testes adicionais
└── requirements.txt         # Dependências do projeto
```

## ⚙️ Funcionalidades

### `extratorFeatures.py`
Contém duas funções principais:
- `extrair_contornos(img, paletaFogo=False, AreaMin=100)`:  
  Detecta regiões com fumaça ou fogo a partir de uma imagem. Utiliza máscara de cor HSV para identificar áreas relevantes.
- `extrair_features(img_hsv, contorno)`:  
  Extrai estatísticas de textura (entropia, média, moda, desvio padrão, mediana) das regiões detectadas.

### `main.ipynb`
Executa o pipeline completo:
- Carregamento de vídeos.
- Extração de frames.
- Processamento de imagens com `extratorFeatures.py`.
- Geração e armazenamento de datasets com características visuais.

### `dataset.ipynb`
- Carrega e explora os datasets gerados.
- É possível anotar o dataset manualmente para cada um dos contornos detectados
  - Por ser um trabalho demorado é possivel interromper o processo e continuar de onde parou a qualquer momento

### `outros.ipynb`
- Arquivo auxiliar para testes, possibilitando a extração de frames de um vídeo, selecionar frames aleatorios para o treinamento e ver os contornos de um vídeo sem classificação.
