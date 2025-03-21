import streamlit as st
import cv2
import time
import numpy as np
import uuid
import os
import tempfile
import subprocess
from pathlib import Path
import threading
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from groq import Groq

# Set page config
st.set_page_config(
    page_title="Video Analysis Suite",
    page_icon="🎥",
    layout="wide"
)

# Main title
st.title("Video Analysis Suite")

# Create tabs for the two parts
tab1, tab2 = st.tabs(["Violence Detection & Audio Analysis", "Vehicle Speed Detection"])

# Part 1: Violence Detection and Audio Extraction
with tab1:
    st.header("Violence Detection with Audio Extraction")
    
    # Violence Detector Class
    class ViolenceDetector:
        def __init__(self, model_path, confidence_threshold=0.7, skip_frames=3):
            """
            Initialize the violence detector with a trained model
            """
            self.model = load_model(model_path)
            self.confidence_threshold = confidence_threshold
            self.skip_frames = skip_frames
            self.frame_count = 0
            self.current_prediction = "Non-Violence"
            self.prediction_confidence = 0.0
            self.processing_frame = False
            self.lock = threading.Lock()
            
        def preprocess_frame(self, frame):
            """Preprocess a single frame for model input"""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (128, 128))
            normalized_frame = resized_frame.astype('float32') / 255.0
            return np.expand_dims(normalized_frame, axis=0)
        
        def predict_frame(self, frame):
            """Process a frame and make prediction"""
            if self.processing_frame:
                return
                
            self.processing_frame = True
            
            processed_frame = self.preprocess_frame(frame)
            prediction = self.model.predict(processed_frame, verbose=0)[0][0]
            
            with self.lock:
                self.prediction_confidence = float(prediction)
                self.current_prediction = "Violence" if prediction >= self.confidence_threshold else "Non-Violence"
            self.processing_frame = False
        
        def process_frame(self, frame):
            """Process a frame and return the labeled frame"""
            if self.frame_count % self.skip_frames == 0 and not self.processing_frame:
                self.predict_frame(frame)
                
            self.frame_count += 1
            
            display_frame = frame.copy()
            
            with self.lock:
                prediction = self.current_prediction
                confidence = self.prediction_confidence
                
            if prediction == "Violence":
                color = (255, 0, 0)  # Red for violence (RGB for Streamlit)
            else:
                color = (0, 255, 0)  # Green for non-violence
                
            # Convert frame to RGB for Streamlit
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            return display_frame_rgb, prediction, confidence, color

    # Media Processor Class
    class MediaProcessor:
        def __init__(self, api_key=None):
            """Initialize with Groq API key."""
            self.api_key = api_key or os.environ.get("GROQ_API_KEY", "put_yo_api_key_here")
            self.client = Groq(api_key=self.api_key)
        
        def extract_audio_from_video(self, video_path, output_audio_path=None):
            """Extract audio from video file using FFmpeg."""
            video_path = Path(video_path)
            
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if output_audio_path is None:
                output_audio_path = video_path.with_suffix('.wav')
            
            try:
                cmd = [
                    'ffmpeg', 
                    '-i', str(video_path), 
                    '-q:a', '0',
                    '-map', 'a',
                    '-y',
                    str(output_audio_path)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_audio_path
            except subprocess.CalledProcessError as e:
                raise Exception(f"Error extracting audio: {e.stderr.decode()}")
            except FileNotFoundError:
                raise Exception("FFmpeg not found. Please install FFmpeg to process video files.")
        
        def transcribe_audio(self, audio_path, options=None):
            """Transcribe audio using Groq's API."""
            audio_path = Path(audio_path)
            
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            default_options = {
                "model": "whisper-large-v3-turbo",
                "response_format": "json",
                "temperature": 0.0,
                "language": "en",
                "prompt": "Specify context or spelling, include emotion indicators. If no words are detected then just return empty string"
            }
            
            if options:
                default_options.update(options)
            
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(str(audio_path), file.read()),
                    **default_options
                )
            
            return transcription.text
        
        def process_video_file(self, video_path, options=None, output_dir="output", keep_audio=False):
            """Process a video file by extracting audio and transcribing it."""
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = Path(video_path).stem
            
            if keep_audio:
                audio_path = Path(output_dir) / f"{base_name}.wav"
            else:
                temp_dir = tempfile.gettempdir()
                audio_path = Path(temp_dir) / f"{base_name}_{os.urandom(4).hex()}.wav"
            
            self.extract_audio_from_video(video_path, audio_path)
            
            try:
                transcription_text = self.transcribe_audio(audio_path, options)
                
                if not keep_audio and temp_dir in str(audio_path):
                    audio_path.unlink(missing_ok=True)
                    
                return transcription_text
            except Exception as e:
                if not keep_audio and temp_dir in str(audio_path):
                    audio_path.unlink(missing_ok=True)
                raise e
    
    # Input selection
    source_option = st.radio(
        "Select video source:",
        ["Upload video file", "Use webcam"]
    )
    
    if source_option == "Upload video file":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "webm"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            temp_file.close()
            
            # Process button
            if st.button("Process Video"):
                try:
                    # Initialize detector with a pretrained model path
                    # Note: In production, replace with actual model path
                    model_path = "violence_detection_model.h5"
                    
                    # Display a warning if model doesn't exist (for demo purposes)
                    if not os.path.exists(model_path):
                        st.warning(f"Model file '{model_path}' not found. Using a placeholder for demo.")
                        # Create a placeholder detector that alternates predictions for demonstration
                        class PlaceholderDetector:
                            def __init__(self):
                                self.frame_count = 0
                            
                            def process_frame(self, frame):
                                self.frame_count += 1
                                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                prediction = "Violence" if self.frame_count % 60 > 30 else "Non-Violence"
                                confidence = 0.7 if prediction == "Violence" else 0.9
                                color = (255, 0, 0) if prediction == "Violence" else (0, 255, 0)
                                return display_frame, prediction, confidence, color
                        
                        detector = PlaceholderDetector()
                    else:
                        detector = ViolenceDetector(
                            model_path=model_path,
                            confidence_threshold=0.7,
                            skip_frames=3
                        )
                    
                    # Create processor for audio extraction
                    processor = MediaProcessor()
                    
                    # Set up video processing
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error opening video file")
                    else:
                        # Create placeholder for video
                        video_placeholder = st.empty()
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Get video properties
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Process video frames
                        frame_count = 0
                        violence_detected = False
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process the frame
                            processed_frame, prediction, confidence, color = detector.process_frame(frame)
                            
                            # Check if violence is detected
                            if prediction == "Violence" and confidence > 0.7:
                                violence_detected = True
                            
                            # Display frame with prediction
                            video_placeholder.image(processed_frame, caption=f"{prediction}: {confidence:.2f}", use_column_width=True)
                            
                            # Update progress
                            frame_count += 1
                            progress = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_placeholder.text(f"Processing frame {frame_count}/{total_frames}")
                            
                            # Control playback speed
                            time.sleep(1/fps)
                        
                        cap.release()
                        
                        # Process audio extraction and transcription
                        st.subheader("Audio Transcription")
                        with st.spinner("Extracting and transcribing audio..."):
                            try:
                                transcription = processor.process_video_file(video_path)
                                st.success("Audio processing completed!")
                                st.text_area("Transcription", transcription, height=200)
                            except Exception as e:
                                st.error(f"Error processing audio: {str(e)}")
                        
                        # Summary
                        # st.subheader("Analysis Summary")
                        # if violence_detected:
                        #     st.error("⚠️ Violence was detected in this video")
                        # else:
                        #     st.success("✅ No violence detected in this video")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(video_path):
                        os.unlink(video_path)
    
    else:  # Webcam option
        st.warning("Note: Webcam capture requires camera permissions")
        
        if st.button("Start Webcam Analysis"):
            try:
                # Initialize detector
                model_path = "violence_detection_model.h5"
                
                # Display a warning if model doesn't exist (for demo purposes)
                if not os.path.exists(model_path):
                    st.warning(f"Model file '{model_path}' not found. Using a placeholder for demo.")
                    class PlaceholderDetector:
                        def __init__(self):
                            self.frame_count = 0
                        
                        def process_frame(self, frame):
                            self.frame_count += 1
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            prediction = "Non-Violence"
                            confidence = 0.95
                            color = (0, 255, 0)
                            return display_frame, prediction, confidence, color
                    
                    detector = PlaceholderDetector()
                else:
                    detector = ViolenceDetector(
                        model_path=model_path,
                        confidence_threshold=0.7,
                        skip_frames=3
                    )
                
                # Set up webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam")
                else:
                    # Create placeholder for video
                    video_placeholder = st.empty()
                    stop_button = st.button("Stop")
                    
                    while not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to capture frame from webcam")
                            break
                        
                        # Process the frame
                        processed_frame, prediction, confidence, color = detector.process_frame(frame)
                        
                        # Display frame with prediction
                        video_placeholder.image(processed_frame, caption=f"{prediction}: {confidence:.2f}", use_column_width=True)
                        
                        # Check if stop button pressed
                        stop_button = st.button("Stop", key="stop_webcam")
                        if stop_button:
                            break
                        
                        # Add small delay to reduce CPU usage
                        time.sleep(0.1)
                    
                    cap.release()
                    st.success("Webcam analysis stopped")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Part 2: Speed Detection
with tab2:
    st.header("Vehicle Speed Detection")
    
    # Display a sample video analysis (hardcoded as specified)
    st.write("Analyzing pre-recorded traffic video...")
    
    # Function for speed detection
    def run_speed_detection():
        # Load YOLO model
        try:
            model = YOLO('yolov8n.pt')  # Use pretrained model
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            st.info("To use this feature, install the required packages with: pip install ultralytics")
            return
        
        # Get sample video path
        video_path = '1.webm'  # This would be your hardcoded video
        if not os.path.exists(video_path):
            st.warning(f"Sample video '{video_path}' not found. Using a placeholder video for demonstration.")
            # Create a placeholder video frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Sample Traffic Video", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display placeholder image
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Placeholder for speed detection video", use_column_width=True)
            return
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video: {video_path}")
            return
            
        # Store vehicle positions and IDs
        vehicle_positions = {}
        vehicle_ids = {}
        speed_threshold = 60  # Speed limit in km/h
        FPS = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        PIXEL_TO_METER = 0.05  # Scale factor (depends on camera position)
        
        # Vehicle classes we're interested in
        VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']
        
        # Generate unique ID for each new vehicle
        def get_vehicle_id(x, y):
            for vid, (vx, vy) in vehicle_positions.items():
                if np.sqrt((vx - x) ** 2 + (vy - y) ** 2) < 50:  # Threshold to consider same object
                    return vid
            new_id = str(uuid.uuid4())
            return new_id
        
        # Create frame placeholder
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Get total frames for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Dictionary to track speed history
        speed_history = {}
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            
            # Run YOLO model on the frame
            results = model(frame)
            
            new_positions = {}
            current_vehicles = 0
            speeding_vehicles = 0
            
            for r in results:
                boxes = r.boxes
                for i, box in enumerate(boxes.xyxy):  # xyxy: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box[:4].int().tolist()
                    
                    # Get class index and confidence
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    
                    # Get class name from the model's names dictionary
                    cls_name = model.names[cls_id]
                    
                    # Only process if it's a vehicle
                    if cls_name in VEHICLE_CLASSES:
                        current_vehicles += 1
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Get or assign vehicle ID
                        object_id = get_vehicle_id(center_x, center_y)
                        new_positions[object_id] = (center_x, center_y)
                        
                        # Calculate speed if previous position exists
                        if object_id in vehicle_positions:
                            prev_x, prev_y = vehicle_positions[object_id]
                            dist_moved = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) * PIXEL_TO_METER
                            time_elapsed = 1 / FPS
                            speed = (dist_moved / time_elapsed) * 3.6  # m/s to km/h
                            
                            # Store speed for display
                            vehicle_ids[object_id] = speed
                            
                            # Track speed history
                            if object_id not in speed_history:
                                speed_history[object_id] = []
                            
                            speed_history[object_id].append(speed)
                            
                            # Get average speed over the last few frames
                            avg_speed = sum(speed_history[object_id][-5:]) / min(5, len(speed_history[object_id]))
                            
                            # Check if speeding
                            is_speeding = avg_speed > speed_threshold
                            if is_speeding:
                                speeding_vehicles += 1
                            
                            # Choose color based on speed threshold
                            color = (0, 255, 0) if not is_speeding else (0, 0, 255)  # Green for normal, Red for overspeed
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Display vehicle type above box
                            cv2.putText(frame, f"{cls_name.upper()}", (x1, y1 - 25),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                            # Display speed below vehicle type
                            cv2.putText(frame, f"{avg_speed:.2f} km/h", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            # If no previous position, assign white box with just the vehicle type
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(frame, f"{cls_name.upper()}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Update positions
            vehicle_positions = new_positions
            
            # Add info text to frame
            cv2.putText(frame, f"Speed limit: {speed_threshold} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Vehicles: {current_vehicles}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Speeding: {speeding_vehicles}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert to RGB for Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            frame_placeholder.image(rgb_frame, caption="Vehicle Speed Detection", use_column_width=True)
            
            # Display stats
            stats_placeholder.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Slow down playback a bit for better viewing
            time.sleep(0.05)
            
        cap.release()
        st.success("Video processing complete!")
        
        # Final summary
        st.subheader("Speed Detection Summary")
        unique_vehicles = len(speed_history)
        max_speed = max([max(speeds) if speeds else 0 for speeds in speed_history.values()]) if speed_history else 0
        
        st.write(f"- Total unique vehicles detected: {unique_vehicles}")
        st.write(f"- Maximum speed recorded: {max_speed:.2f} km/h")
        st.write(f"- Speed limit: {speed_threshold} km/h")
        
        # Display speed distribution chart if there's data
        if speed_history:
            avg_speeds = [sum(speeds)/len(speeds) for speeds in speed_history.values() if speeds]
            speed_ranges = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "80+": 0}
            
            for speed in avg_speeds:
                if speed <= 20:
                    speed_ranges["0-20"] += 1
                elif speed <= 40:
                    speed_ranges["21-40"] += 1
                elif speed <= 60:
                    speed_ranges["41-60"] += 1
                elif speed <= 80:
                    speed_ranges["61-80"] += 1
                else:
                    speed_ranges["80+"] += 1
            
            st.bar_chart(speed_ranges)
    
    # Run the speed detection function
    if st.button("Run Speed Detection"):
        run_speed_detection()
    else:
        st.info("Click 'Run Speed Detection' to analyze the pre-recorded traffic video")