mkdir dataset
# Download train, validation, test datasets using wget
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip?download=true -O dataset/train.zip -c
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip?download=true -O dataset/val.zip -c
wget https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_test.zip?download=true -O dataset/test.zip -c
#widerfice annotation
wget http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip -O dataset//wider_face_split.zip -c