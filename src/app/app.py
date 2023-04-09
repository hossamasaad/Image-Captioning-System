import streamlit as st


# General settings
st.title("Image Captioning ")
submit = None


# File upload
uploading = st.container()

upload_columns = uploading.columns([2, 1])
uploaded_file = upload_columns[0].file_uploader(
    label="Upload an Image",
    type=['png', 'jpg'],
    accept_multiple_files=False
)

if uploaded_file:
    upload_columns[1].image(uploaded_file)

st.markdown("""---""")


# Choosing Model
if uploaded_file:
    submitting = st.container()
    model_type = submitting.selectbox('', ["Model 1", "Model 2", "Model 3"])
    submit = submitting.button("Caption it!")
    st.markdown("""---""")


# Generation
generating = st.container()
if submit:
    generating.write(
        """
        <style type="text/css">
            .generated-text {
                font-family:Courier, monospace;
                font-size:16px;
                line-height:20px;
                text-align:center;
                color:#ffffff;
                background-color:#7e7bb0;
                padding:20px;
            } 
        </style>
        <p class="generated-text">A handsome guy with cute smile</p>
        """,
    unsafe_allow_html=True)