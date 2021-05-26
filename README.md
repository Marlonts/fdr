# Detecção e Reconhecimento de Faces

- Este projeto foi desenvolvido como parte integrante da avaliação 
da disciplina INE - Visão Computacional - PPGCC UFSC

# -----------------------------------------------------------------

## Primeira Etapa - Detecção de pessoas

### Visão compuacional classica (Classical Computer Vision - CCV)

  $ python3 detect_ccv.py -i images/1.jpg

### Rede Neural Convolucional (Convolutional Neural Network - CNN)

  $ python3 detect_cnn.py --config-file ../../../../detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml \
    --input images/1.jpg \
    --opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl

# -----------------------------------------------------------------

## Segunda Etapa - Reconhecimento Facial

### Detecção de faces

#### Datasets
Faça download e extraía o dataset de faces pré-delimitadas para
a pasta facial_recognition/data/images/detect_images
https://www.kaggle.com/vin1234/count-the-number-of-faces-present-in-an-image

Faça download e extraía o dataset de faces de celebridades para 
a pasta facial_recognition/data/images/recog_images
https://www.kaggle.com/atulanandjha/lfwpeople

#### Prepare os modelos

##### Pŕe-processamento dos dados

    $ python3 preprocess_data.py

#### Treinamento
    
    $ python3 train.py


### Classificação de faces

### Registre a face

    # Adicione uma ou mais fotos da pessoa a ser recohecida em images/recog_images/<name> folder

    $ python3 register_recog_images.py

### Execute o modelo - Imagem

    $ python3 recog.py data/images/test_images/1.jpg
