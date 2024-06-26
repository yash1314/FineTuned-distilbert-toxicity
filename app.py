import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("Yash907/ft-distb-toxicity")
    model = AutoModelForSequenceClassification.from_pretrained("Yash907/ft-distb-toxicity")
    return tokenizer,model

tokenizer, model = get_model()

st.title(':rainbow[Fine-Tuned Sentiment Analysis]')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

st.markdown("""This webapp can check whether your input sentence is toxic or non-toxic. The model used
here is fine-tuned on custom dataset [link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/models) for detecting toxicity.""", 
unsafe_allow_html=True)

input_text = st.text_area(label = ' ', placeholder = 'Enter your text here', max_chars = 512, 
                     label_visibility = "collapsed")
button = st.button('Analyze')

toxicity = {
    1:'Toxic',
    0:'Non-Toxic'
}

if input_text and button:
    
    tokenized_input = tokenizer([input_text], padding=True, truncation=True, max_length=512,return_tensors='pt')
    
    output = model(**tokenized_input)

    y_pred = np.argmax(output.logits.detach().numpy(), axis = 1)

    st.success(f"Prediction:  {toxicity[y_pred[0]]}")    