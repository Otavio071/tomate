# 🍅 Detecção de Doenças Foliares em Tomateiros com Visão Computacional Explicável (XAI)

Classificação de doenças em folhas de tomateiro com **transfer learning (DenseNet121)** e interpretação das predições com **Grad-CAM**, para mostrar quais regiões da folha o modelo usa para decidir.

Projeto da disciplina de **Fundamentos da Inteligência Artificial** (Ciência da Computação, UFS), com foco em aplicações de IA para o agronegócio.

## Objetivo

Dada uma imagem de folha de tomateiro, classificá-la em uma de 10 classes (doenças ou folha saudável) e **explicar visualmente** a decisão do modelo. A explicabilidade é importante em aplicações agrícolas: quem usa o sistema precisa confiar que o modelo olha para a lesão da folha e não para o fundo da imagem.

## Dataset

- [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset), subconjunto de tomate, com **10 classes** (doenças e folhas saudáveis)
- **10.000 imagens** de treino e **1.000 imagens** de validação
- Imagens redimensionadas para 256×256 e normalizadas para o intervalo [0, 1]

Estrutura esperada pelo notebook:

```
tomato/
├── train/
│   ├── Tomato___Bacterial_spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   └── ... (10 pastas, uma por classe)
└── val/
    └── ... (mesmas 10 pastas)
```

## Modelo

| Item | Configuração |
|---|---|
| Base | DenseNet121 pré-treinada na ImageNet, congelada (`include_top=False`, `pooling='avg'`) |
| Cabeça de classificação | BatchNormalization → Dense(256, ReLU) → Dropout(0,35) → BatchNormalization → Dense(128, ReLU) → Dense(10, Softmax) |
| Otimizador | Adam, taxa de aprendizado 1e-4 |
| Loss | Categorical crossentropy |
| Regularização | Dropout e Early Stopping |
| Framework | TensorFlow / Keras |

## Resultados

- **Acurácia de validação: 92,9%**

<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/7355446e-82c9-4173-be03-d0d2363eb1f4" />


## Explicabilidade com Grad-CAM

O Grad-CAM calcula os gradientes da classe prevista em relação aos mapas de ativação da última camada convolucional da DenseNet121 (`conv5_block16_concat`). Com isso, gera um **mapa de calor** que destaca as regiões da folha que mais influenciaram a predição. O notebook mostra, lado a lado:

1. a imagem original da folha;
2. o mapa de calor do Grad-CAM;
3. a sobreposição do mapa de calor na imagem.

## Como executar

1. Abra o notebook `tomato-leaf-gradcam.ipynb` no [Google Colab](https://colab.research.google.com/) (preferencialmente com GPU: *Ambiente de execução → Alterar tipo de ambiente de execução*).
2. Coloque o dataset compactado (`dataset.zip`, com as pastas `train` e `val`) no Google Drive, em `Colab Notebooks/`, ou ajuste o caminho `zip_path` na célula de carregamento dos dados.
3. Execute as células em ordem: carregamento dos dados → treinamento → avaliação → Grad-CAM.

> Algumas células finais usam imagens de teste que ficam no ambiente local do Colab. Para reproduzir, envie suas próprias imagens e ajuste o `img_path`.

## Tecnologias

Python · TensorFlow/Keras · DenseNet121 · Grad-CAM · OpenCV · NumPy · Matplotlib · Google Colab

## Referências

- Wang et al. (2024), *Frontiers in Plant Science*: artigo científico usado como referência na proposta técnica do projeto (detecção de doenças em folhas de tomateiro).
- Dataset PlantVillage.
- Parte do treinamento foi adaptada de um notebook público do Kaggle de Saman Fatima ([kaggle.com/samanfatima7](https://www.kaggle.com/samanfatima7)). A adaptação para o Colab e a implementação do Grad-CAM foram feitas por mim.

## Autor

**Otávio Augusto Batista Santos Filho**, estudante de Ciência da Computação na UFS.
[GitHub](https://github.com/Otavio071) · [LinkedIn](https://linkedin.com/in/otávio-augusto-5b4812436)
