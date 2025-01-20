import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("best.pt")  # Replace 'best.pt' with your trained YOLO model

# Initialize session state for products
if "product" not in st.session_state:
    st.session_state["product"] = []
if "label" not in st.session_state:
    st.session_state["label"] = None

# Streamlit app
st.title("Clothing recognition for inventory management")

# Start/Stop toggle
run = st.checkbox("Start Webcam")
# Streamlit widget for video frame display
stframe = st.empty()
def getProduct():
    if st.session_state["label"]:
        st.session_state["product"].append(st.session_state["label"])
        st.write(f"Detected product added: {st.session_state['label']}")
        st.write(st.session_state["product"])
        
print(st.session_state["label"])
st.button("Detect product", on_click=getProduct)
# Initialize webcam
if run:
    cap = cv2.VideoCapture(0)  # Open webcam feed (index 0)
    
    if not cap.isOpened():
        st.error("Error: Unable to access the webcam.")
        run = False

    while run:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            st.warning("Warning: Unable to read from the webcam.")
            break

        # Run YOLO inference
        results = model(frame)

        # Annotate frame with detection results
        for result in results:
            for i, box in enumerate(result.boxes.xyxy):  # Process bounding boxes
                if float(result.boxes.conf[i]) > 0.2:
                    
                   
                    # Fetch confidence and class ID
                    try:
                        confidence = float(result.boxes.conf[i])  # Confidence score
                        class_id = int(result.boxes.cls[i])  # Class ID
                    except (IndexError, AttributeError):
                        st.warning(f"Error processing box {i}. Skipping.")
                        continue

                    # Get label from class ID, handle out-of-bounds
                    label  = model.names.get(class_id, f"Unknown({class_id})")
                    st.session_state["label"] = label
                    x1, y1, x2, y2 = map(int, box[:4])  # Coordinates
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)  # Label
                
        # Convert BGR to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # Display video frame
        stframe.image(frame, channels="RGB")
        
    cap.release()  # Release webcam
else:
    st.write("Check the box above to start the webcam.")


