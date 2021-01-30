import streamlit as st
from streamlit_observable import observable

a = st.slider("Số lượng ảnh", value=100)
b = st.slider("Cắt chính xác trường chứa thông tin", value=19)
c = st.slider("Trích xuất thông tin các trường", value=10)

observable("Thống kê độ chính xác hệ thống nhận dạng ảnh cccd", 
    notebook="@juba/updatable-bar-chart", 
    targets=["chart", "draw"], 
    redefine={
        "data": [
            {"name": "Số lượng ảnh", "value": a},
            {"name": "Cắt trường chứa thông tin", "value": b},
            {"name": "Trích xuất thông tin các trường", "value": c}
        ],
    },
    hide=["draw"]
)