!wget https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/project-dog-classification/haarcascades/haarcascade_frontalface_alt.xml -P haarcascades/

!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/American_water_spaniel_00648.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Brittany_02625.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Curly-coated_retriever_03896.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Labrador_retriever_06449.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Labrador_retriever_06455.jpg	 -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Labrador_retriever_06457.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/Welsh_springer_spaniel_08203.jpg -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_cnn.png -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png -P images/
!wget https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_human_output.png -P images/

!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip

!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
!unzip lfw.zip

!nvidia-smi
!pip freeze | grep 'tensor\|torch'