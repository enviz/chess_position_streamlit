import streamlit as st
from keras.models import load_model
import cv2
from skimage import io, transform
from skimage.util.shape import view_as_blocks
import numpy as np
from PIL import Image, ImageOps
 
@st.cache(allow_output_mutation=True)
def load_model_():
  model = load_model("chess_model.h5")
  return model
with st.spinner('Model is being loaded..'):
  model=load_model_()
 
st.write("""
         # Predict Chess position
         """
         )
 
file = st.file_uploader("Upload the image", type=["jpg", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)


def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += tran_t(one_hot[j][i])
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

SQUARE_SIZE = 40 # must be less than 400/8 = 50
downsample_size = SQUARE_SIZE * 8
square_size = SQUARE_SIZE

def split_chessboard_into_64_images(image):
    
    #img_read = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    img_read = np.array(image)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size))
    
    tiles =  tiles.reshape(64, square_size, square_size)
    
    return tiles.tolist()


def tran(t):
    T={'B':0,'b':1,'K':2,'k':3,'Q':4,'q':5,'R':6,'r':7,'P':8,'p':9,'N':10,'n':11,'F':12}
    return T[t]

def tran_t(t):
    T={0:'B',1:'b',2:'K',3:'k',4:'Q',5:'q',6:'R',7:'r',8:'P',9:'p',10:'N',11:'n'}
    return T[t]
 
def upload_predict(upload_image, model):


        #image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        #image = np.asarray(image)
        
        test_input_image = split_chessboard_into_64_images(upload_image)
        test_input_image = np.array(test_input_image)
        test_input_image = test_input_image.reshape(test_input_image.shape + (1,))
        pred = model.predict(test_input_image)
        pred = pred.argmax(axis=1).reshape(-1, 8, 8)
        
        return pred
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_gray = ImageOps.grayscale(image)
    pred = upload_predict(image_gray, model)
    image_label = str(fen_from_onehot(pred[0]))
    
    st.write("The predicted FEN:",image_label)
    st.write('True Label:', file.name.replace('.jpeg',''))
