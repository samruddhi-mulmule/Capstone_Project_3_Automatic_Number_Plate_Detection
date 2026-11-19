import os
import pickle

import cv2
import numpy as np
import streamlit as st

MODEL_PATH = "best_model.pkl"                 # saved from your notebook
CASCADE_PATH = "indian_license_plate.xml"     # from the Kaggle dataset zip
CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # same order as the class_0 ... class_Z folders


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)   # some OpenCV versions raise on a bad file
    if cascade.empty():                             # others return an empty classifier
        raise ValueError("cascade file could not be loaded")
    return cascade


# ---------------------------------------------------------------- step 1: find the plate
def find_plate(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # try strict detection first, then relax if nothing is found
    for n in [7, 5, 3]:
        plates = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=n)
        if len(plates) > 0:
            break

    if len(plates) == 0:
        return None, None

    # keep only the biggest detection
    x, y, w, h = max(plates, key=lambda p: p[2] * p[3])

    plate = img[y:y + h, x:x + w].copy()

    detected = img.copy()
    cv2.rectangle(detected, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return detected, plate


# ---------------------------------------------------------------- step 2: cut the plate into characters
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lower_width, upper_width, lower_height, upper_height = dimensions

    # check the 15 largest contours
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    x_list = []
    img_res = []

    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)

        # keep only contours that are the size of a character
        if lower_width < w < upper_width and lower_height < h < upper_height:
            x_list.append(x)

            char = img[y:y + h, x:x + w]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)            # invert: white character on black

            char_copy = np.zeros((44, 24))
            char_copy[2:42, 2:22] = char              # black border, same as training images
            img_res.append(char_copy)

    # left-most character first
    order = sorted(range(len(x_list)), key=lambda k: x_list[k])
    return np.array([img_res[i] for i in order])


def segment_characters(plate):
    img_lp = cv2.resize(plate, (333, 75))
    gray = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)                     # smooths watermark specks and noise
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:                                      # light text on dark plate -> flip
        binary = 255 - binary
    binary = cv2.erode(binary, (3, 3))
    binary = cv2.dilate(binary, (3, 3))

    lp_width = binary.shape[0]
    lp_height = binary.shape[1]

    # make borders white
    binary[0:3, :] = 255
    binary[:, 0:3] = 255
    binary[72:75, :] = 255
    binary[:, 330:333] = 255

    dimensions = [lp_width / 6, lp_width / 2, lp_height / 10, 2 * lp_height / 3]

    return find_contours(dimensions, binary), binary


# ---------------------------------------------------------------- step 3: read the characters
def read_characters(char_list, model):
    batch = []

    for ch in char_list:
        ch = cv2.resize(ch.astype("float32"), (64, 64), interpolation=cv2.INTER_NEAREST)
        ch = np.stack([ch, ch, ch], axis=-1) / 255.0      # 3 channels + same /255 as training
        batch.append(ch)

    pred = model.predict(np.array(batch), verbose=0)

    return "".join(CHARACTERS[i] for i in pred.argmax(axis=1))


# ---------------------------------------------------------------- the web page
def main():
    st.set_page_config(page_title="Number Plate Recognition", page_icon="🚗")
    st.title("🚗 Indian Number Plate Recognition")
    st.write("Upload a photo of a car. The app finds the number plate and reads its letters and numbers.")

    # check the two files are next to app.py
    missing = [p for p in (MODEL_PATH, CASCADE_PATH) if not os.path.exists(p)]
    if missing:
        st.error("Missing file(s): " + ", ".join(missing) + ". Put them in the same folder as app.py.")
        st.stop()

    if not hasattr(cv2, "CascadeClassifier"):
        st.error("Your OpenCV version (" + cv2.__version__ + ") does not include CascadeClassifier. "
                 "Install the version used in the notebook:  pip install opencv-python==4.10.0.84")
        st.stop()

    try:
        cascade = load_cascade()
    except Exception as e:
        with open(CASCADE_PATH, "rb") as f:
            start = f.read(60)
        st.error(CASCADE_PATH + " could not be loaded. Use the original file from the Kaggle dataset.")
        st.code(
            "File size (bytes): " + str(os.path.getsize(CASCADE_PATH)) + "\n"
            "File starts with : " + repr(start) + "\n"
            "OpenCV version   : " + cv2.__version__ + "\n"
            "Error            : " + repr(e)
        )
        st.stop()

    model = load_model()

    file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png", "webp"])
    if file is None:
        st.stop()

    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("This file could not be read as an image.")
        st.stop()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # shrink very large photos so the plate finder works better
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)

    # show the uploaded image and wait for the button
    st.image(img, caption="Uploaded image")
    if not st.button("Detect Number Plate", type="primary"):
        st.stop()

    detected, plate = find_plate(img, cascade)
    if plate is None:
        st.image(img, caption="Uploaded image")
        st.error("No number plate found. Try a clearer, front-facing photo of the plate.")
        st.stop()

    char_list, binary = segment_characters(plate)
    if len(char_list) == 0:
        st.image(detected, caption="Detected plate")
        st.error("Plate found, but the characters could not be separated.")
        st.stop()

    text = read_characters(char_list, model)

    st.image(detected, caption="Detected plate")
    st.subheader("Plate number: " + text)

    with st.expander("See the steps"):
        st.write("Cropped plate")
        st.image(plate)
        st.write("Black and white version")
        st.image(binary)
        st.write("Characters sent to the model")
        st.image([c.astype("uint8") for c in char_list], width=50)


if __name__ == "__main__":
    main()