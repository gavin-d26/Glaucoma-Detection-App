import streamlit as st
import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import PIL 
import os
import configs
import app_utils
import ensemble as ensemble

@st.cache
def get_model_path_df():
    df = pd.read_csv(configs.MODEL_PATHS_DF, index_col= 'index')
    return df


@st.cache
def load_ensemble(model_list, df, saved_models_path, reduction='max'):
    model = ensemble.Ensemble(model_list, df, saved_models_path, reduction=reduction)
    model.eval()
    return model

def set_use_preload_True():
    st.session_state.use_preloaded = True
    
def set_use_preload_False():
    st.session_state.use_preloaded = False


def get_uploaded_file():
    st.session_state.get_uploaded_file = True

def set_model_exists_true():
    st.session_state.model_exists = True

def set_model_exists_false():
    st.session_state.model_exists = False
    
def check_model_list():
    if len(st.session_state.model_list)==0:
        st.session_state.model_exists = False

def clear_uploads_fn():
    if len(os.listdir(configs.UPLOADS_DIR_PATH)) > 0:
        for file_name in os.listdir(configs.UPLOADS_DIR_PATH):
            os.remove(os.path.join(configs.UPLOADS_DIR_PATH, file_name))
        
    

if "use_preloaded" not in st.session_state:
    st.session_state.use_preloaded = True

if "curr_img" not in st.session_state:
    st.session_state.curr_img = None
    
if "get_uploaded_file" not in st.session_state:
    st.session_state.get_uploaded_file = False
    
if "model_exists" not in st.session_state:
    st.session_state.model_exists = False


if st.session_state.model_exists is False:
    model = None

# st.session_state

if __name__=="__main__":
    
    st.title("Glaucoma Detector")
    st.write("***")
    
    st.subheader("1) Create Model")
    
    df = get_model_path_df()

    model_list = st.multiselect("Select Models for Ensemble", df.index.tolist(), None, key="model_list", on_change= set_model_exists_false)
    #st.text("")
    col1, col2, col3 = st.columns(3)
    with col1:
        reduction = st.select_slider("Select Reduction Scheme", ['max', 'mean'], key="reduction", on_change=set_model_exists_false)
        
    build_model_button = st.button("Build", key='build_model') #on_click=set_model_exists_true)
    
    # st.write(type(reduction))####
    
    if len(st.session_state.model_list)==0: 
        st.session_state.model_exists=False
        build_model_button = False
    
    if build_model_button or (st.session_state.model_exists is True): #st.session_state.build_model:
        #st.write(reduction)###
        model = load_ensemble(st.session_state.model_list, df, configs.SAVED_MODELS_PATH, reduction=st.session_state.reduction)
        #st.write(model)
        st.session_state.model_exists = True
        # st.write("Model Built!")
        st.markdown('<p style="color:Green">Model Built!</p>', unsafe_allow_html=True)
        
    else:
        st.markdown('<p style="color:Red">Model Not Built!</p>', unsafe_allow_html=True)
        # st.write("1)Choose Models for ensemble")
        # st.write("2)Choose Reduction Scheme")
    
    st.markdown("***")
    st.subheader("2) Choose Image")
    
    files = st.file_uploader("Upload Image", ['png', 'jpg'], accept_multiple_files=True, on_change = get_uploaded_file)
 
    if len(files) == 0:
        st.session_state.get_uploaded_file = False
    
    
    if(len(files) !=0) and (st.session_state.get_uploaded_file is True):
        for file in files:
            img = app_utils.image_format(file)
            img.save(os.path.join(configs.UPLOADS_DIR_PATH, file.name))
                     
        st.session_state.use_preloaded = False
        st.session_state.curr_uploaded_img_id = files[0].name
        st.session_state.get_uploaded_file = False
        
    curr_preloaded_img_id = st.sidebar.selectbox("Preloaded Images", os.listdir(configs.PRELOADED_DIR_PATH), key="curr_preloaded_img_id", on_change=set_use_preload_True)
    curr_uploaded_img_id = st.sidebar.selectbox("Uploaded Images", os.listdir(configs.UPLOADS_DIR_PATH), key="curr_uploaded_img_id", on_change=set_use_preload_False)
    

    
    ####
    
    st.sidebar.button("clear uploads", key="clear_uploads", on_click=clear_uploads_fn)
    
    
    
    ####
    if curr_uploaded_img_id is None:
        st.session_state.use_preloaded = True
    
    
    if (st.session_state.use_preloaded is True): #or (st.session_state.use_preloaded is None):
        st.session_state.curr_img = PIL.Image.open(os.path.join(configs.PRELOADED_DIR_PATH, curr_preloaded_img_id))
        st.session_state.curr_imgid = curr_preloaded_img_id
    else:
        st.session_state.curr_img = PIL.Image.open(os.path.join(configs.UPLOADS_DIR_PATH, curr_uploaded_img_id))
        st.session_state.curr_imgid = curr_uploaded_img_id
    
    # st.text("") 
    # st.text("")   
    st.write("***")
    st.subheader("3) See Result")
    col4, col5, col6 = st.columns(3)
    with col4:
        # st.text("Image ID: {}".format(st.session_state.curr_imgid))   
        st.markdown('<p><span>Image ID: &nbsp </span><span style="background-color: lightblue"><b>{}</b></span></p>'.format(st.session_state.curr_imgid), unsafe_allow_html=True)
        st.image(st.session_state.curr_img)
    
    
    image = app_utils.preprocess(st.session_state.curr_img)
    # st.text("")
    # st.text("")
    run_model = st.button("Run")
    
    if run_model is True:
        if model is None:
            # st.write("Build model First!")
            with col5:
                st.markdown("<p><br/><br/><br/><br/><br/></p>", unsafe_allow_html=True)    
                st.markdown('<p style="color:red"><span>&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp </span>Build a Model First!</p>',unsafe_allow_html=True)
        else:
            out, heatmap = model.predict(image)
            out_class = 'RG' if out>0.5 else 'NRG'
            out=round(out*100, 2)
            overlayed_img = app_utils.hmap_post_process(heatmap, st.session_state.curr_img)
            
            
            with col5:
                
                st.write('<p>&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp <span style="color:darkorange"><b>HeatMap</b></span></p>', unsafe_allow_html=True)
                st.image(overlayed_img)
            
            with col6:
                st.markdown("<p><br/><br/><br/><br/><br/></p>", unsafe_allow_html=True)
                if out_class=='RG':
                    st.markdown('<p>Class: <span style="color:red"><b>{}</b></span></p>'.format(out_class), unsafe_allow_html=True) 
                    st.markdown('<p>Output: <span style="color:red"><b>{}</b></span>%</p>'.format(out), unsafe_allow_html=True)
                else:
                    st.markdown('<p>Class: <span style="color:green"><b>{}</b></span></p>'.format(out_class), unsafe_allow_html=True) 
                    st.markdown('<p>Output: <span style="color:green"><b>{}</b></span>%</p>'.format(out), unsafe_allow_html=True)    
                # st.markdown(outtext)
                # st.write(f'{out*100:.2f}%')
            # conc = "RG" if out>0.5 else "NRG"
            # st.text("Class: {}".format(conc))
            # st.text("")
            # st.text("Heatmap")
            # st.image(heatmap)
    
    # st.session_state
   



    
    