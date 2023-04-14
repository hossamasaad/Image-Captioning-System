import requests
import streamlit as st


# -------------- General settings ----------------

st.title("Image Captioning ")
submit = None

# --------------- File uploading ------------------

uploading = st.container()
upload_columns = uploading.columns([2, 1])
uploaded_file = upload_columns[0].file_uploader(
    label="Upload an Image",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False
)

if uploaded_file:
    upload_columns[1].image(uploaded_file)
st.markdown("""---""")


# ----------------- Choosing Model -----------------

if uploaded_file:
    submitting = st.container()
    model_type = submitting.selectbox('', ["vit-gpt2", "Model 2", "Model 3"])
    submit = submitting.button("Caption it!")
    st.markdown("""---""")



# ----------------- Generation ----------------------

generating = st.container()
if submit:

    # Save uploaded image
    with open(f"saved_images/{uploaded_file.name}", "wb") as file:
        file.write(uploaded_file.getbuffer())
        
    # make request
    caption = ""
    data = {
        "image_path": "src/app/saved_images/" + uploaded_file.name
    }

    # Send request
    if model_type == "vit-gpt2":
        response = requests.post(
            "http://127.0.0.1:8000/vit",
            json=data
        )

        caption = response.json()["caption"][0]
    else:
        caption = "Not implementeed yet"
    
    # Caption style
    html = """
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
    """
    html += f'<p class="generated-text">{caption}</p>' 
    generating.write(html, unsafe_allow_html=True)