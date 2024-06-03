import streamlit as st
import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from detect_pneumonia import predict_pneumonia_resnet18, predict_pneumonia_resnet101, predict_pneumonia_densenet121
from detect_heart import detect_heart_resnet18

def make_pred(architecture, x_ray):
    device = torch.device("cpu")

    x_ray = np.asarray(x_ray) / 255
    x_ray = cv2.resize(x_ray, (224, 224)).astype(np.float32)

    pneumonia_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.49044,], [0.24787,])
    ])
    heart_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.49466,], [0.25284,])
    ])

    x_ray_pneumonia = pneumonia_transform(x_ray)
    if architecture == "ResNet101":
        activation_map, pred = predict_pneumonia_resnet101(device, x_ray_pneumonia)
    elif architecture == "DenseNet121":
        activation_map, pred = predict_pneumonia_densenet121(device, x_ray_pneumonia)
    else:
        activation_map, pred = predict_pneumonia_resnet18(device, x_ray_pneumonia)

    x_ray_heart = heart_transform(x_ray)
    heart_preds = detect_heart_resnet18(device, x_ray_heart)

    return x_ray_pneumonia, activation_map, pred, heart_preds

def create_checkbox(type, visible = False):
    if type == "cam":
        if visible:
            return st.checkbox("Pokaż mapę aktywacji", key='cam1_check')
        else:
            return st.checkbox("Pokaż mapę aktywacji", disabled=True,
                               key='cam2_check')
    else:
        return st.checkbox("Pokaż położenie serca", key='heart_check')

def reset_checkboxes():
    st.session_state.cam1_check = True
    st.session_state.cam2_check = False
    st.session_state.heart_check = True

def main():
    title = "System wykrywający zapalenie płuc i położenie serca na zdjęciu RTG klatki piersiowej"
    st.markdown(f"<h1 style='text-align: center; font-size:3em;'>{title}</h1>",
                unsafe_allow_html=True)
    st.divider()
    left_pane1, right_pane1 = st.columns(2)
    with left_pane1:
        architecture = st.radio("Architektura",
                                ["ResNet18", "ResNet101", "DenseNet121"])
        match architecture:
            case "ResNet18":
                num_par = "11.2M"
                threshold = 0.47
                f1 = 0.907
            case "ResNet101":
                num_par = "42.5M"
                threshold = 0.43
                f1 = 0.888
            case "DenseNet121":
                num_par = "6.9M"
                threshold = 0.48
                f1 = 0.886
    with right_pane1:
        st.write("Dane szczegółowe modelu")
        st.write(f"""Ilość parametrów: {num_par}
                 <br>
                 Próg decyzyjny: {threshold}
                 <br>
                 Miara F1: {f1}""", unsafe_allow_html=True)

    st.divider()
    x_ray = st.file_uploader("Prześlij obraz RTG klatki piersiowej",
                             on_change=reset_checkboxes)
    
    if x_ray:
        x_ray = Image.open(x_ray).convert('L')
        img, heatmap, pred, heart = make_pred(architecture, x_ray)
        
        left_pane2, right_pane2 = st.columns(2)
        with left_pane2:
            if pred.item() > threshold:
                st.subheader(":red[Zdiagnozowano zapalenie płuc]")
                st.write(f"Tensor: {format(pred.item(), '.2f')}")
                cam_state = create_checkbox("cam", True)
            else:
                st.subheader(":green[Nie zdiagnozowano zapalenia płuc]")
                st.write(f"Tensor: {format(pred.item(), '.2f')}")
                cam_state = create_checkbox("cam")
            
            heart_state = create_checkbox("heart")

        with right_pane2:
            img = img[0]
            heatmap = transforms.functional.resize(heatmap.unsqueeze(0),
                                                   (img.shape[0], img.shape[1]))[0]
            
            fig, axis = plt.subplots()
            axis.imshow(img, cmap="bone")

            heart_pos = patches.Rectangle((heart[0], heart[1]), heart[2]-heart[0],
                                        heart[3]-heart[1], linewidth=1, edgecolor='r',
                                        facecolor='none')

            if cam_state:
                axis.imshow(heatmap, alpha=0.3, cmap="jet")
            
            if heart_state:
                axis.add_patch(heart_pos)
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()
