# streamlit_fire_detection.py
import streamlit as st
import torch
import av
import numpy as np
from PIL import Image
import time
from datetime import timedelta
import json
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🔥 X-CLIP Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff6b35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 5px solid #ff6b35;
        background-color: #fff3f3;
    }
    .success-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        background-color: #f3fff3;
    }
    .info-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
        background-color: #f3f9ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None

# Utility functions
def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    td = timedelta(seconds=int(seconds))
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_video_duration(seconds):
    """Format video duration"""
    td = timedelta(seconds=int(seconds))
    if td.seconds >= 3600:
        return f"{td.seconds//3600:02d}:{(td.seconds%3600)//60:02d}:{td.seconds%60:02d}"
    else:
        return f"{td.seconds//60:02d}:{td.seconds%60:02d}"

# Model loading function
@st.cache_resource
def load_xclip_model():
    """Load X-CLIP model with caching"""
    try:
        with st.spinner("🔄 Loading X-CLIP model for first time..."):
            from transformers import AutoProcessor, AutoModel
            
            # Install required packages if not available
            try:
                import torch
                import av
            except ImportError:
                st.error("Required packages not installed. Please install: torch, av, transformers")
                return None, None, None
            
            processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
            # Save locally for offline use
            os.makedirs("./models", exist_ok=True)
            processor.save_pretrained("./models/xclip-processor")
            model.save_pretrained("./models/xclip-model")
            
            return processor, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Detection function
def detect_fire_smoke(frames, processor, model, device, threshold=0.25):
    """Detect fire and smoke using X-CLIP"""
    text_prompts = [
        "a video of actual fire with flames burning",
        "a video of real smoke rising from fire", 
        "a video showing dangerous flames",
        "a video of fire emergency with smoke",
        "a video of building or forest fire",
        "a video of bright lights or LED lighting",
        "a video of red colored objects not on fire",
        "a video of normal indoor or outdoor lighting",
        "a video of normal safe scene without fire or smoke"
    ]
    
    try:
        # Ensure exactly 8 frames
        if len(frames) != 8:
            if len(frames) < 8:
                while len(frames) < 8:
                    frames.extend(frames[:8-len(frames)])
            else:
                step = len(frames) // 8
                frames = [frames[i*step] for i in range(8)]
        
        video_frames = [np.array(frame) for frame in frames]
        
        inputs = processor(
            text=text_prompts,
            videos=[video_frames],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_video = outputs.logits_per_video
            probs = torch.softmax(logits_per_video, dim=-1)
            
            fire_smoke_indices = list(range(0, 5))
            false_positive_indices = list(range(5, 8))
            normal_indices = [8]
            
            fire_smoke_prob = max([probs[0][i].item() for i in fire_smoke_indices])
            false_positive_prob = max([probs[0][i].item() for i in false_positive_indices])
            normal_prob = probs[0][8].item()
            
            is_danger = (
                fire_smoke_prob > threshold and
                fire_smoke_prob > normal_prob and
                fire_smoke_prob >= false_positive_prob
            )
            
            confidence = fire_smoke_prob - max(normal_prob, false_positive_prob)
            
            fire_prob = max([probs[0][0].item(), probs[0][2].item(), probs[0][4].item()])
            smoke_prob = max([probs[0][1].item(), probs[0][3].item()])
            
            return {
                'fire_detected': is_danger,
                'fire_probability': fire_prob,
                'smoke_probability': smoke_prob,
                'fire_smoke_combined': fire_smoke_prob,
                'false_positive_probability': false_positive_prob,
                'normal_probability': normal_prob,
                'detection_confidence': confidence,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None

# Video processing function
def process_video_file(video_file, processor, model, device, threshold=0.25):
    """Process uploaded video file"""
    try:
        # Save uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.getvalue())
        
        container = av.open("temp_video.mp4")
        video_stream = container.streams.video[0]
        total_duration = container.duration / 1000000
        original_fps = float(video_stream.average_rate)
        frame_interval = int(original_fps)
        
        frames_buffer = []
        frame_count = 0
        detection_count = 0
        alerts = []
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame in container.decode(video=0):
            frame_count += 1
            progress = min(frame_count / video_stream.frames, 1.0) if video_stream.frames else 0
            progress_bar.progress(progress)
            
            if frame_count % frame_interval == 0:
                img = frame.to_image()
                img = img.resize((224, 224))
                frames_buffer.append(img)
                
                if len(frames_buffer) >= 8:
                    detection_count += 1
                    current_time = frame_count / original_fps
                    timestamp = seconds_to_timestamp(current_time)
                    
                    status_text.text(f"🔍 Analyzing clip {detection_count} at [{timestamp}]...")
                    
                    results = detect_fire_smoke(
                        frames_buffer[:8], processor, model, device, threshold
                    )
                    
                    if results and results['fire_detected']:
                        alerts.append({
                            'timestamp_seconds': current_time,
                            'timestamp_formatted': timestamp,
                            'confidence': results['detection_confidence'],
                            'details': results
                        })
                    
                    frames_buffer = frames_buffer[-2:]
        
        container.close()
        os.remove("temp_video.mp4")  # Clean up temp file
        
        return alerts, total_duration, detection_count
        
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return [], 0, 0

# RTSP testing function
def test_rtsp_stream(rtsp_url, processor, model, device, max_clips=5, threshold=0.25):
    """Test RTSP stream"""
    try:
        container = av.open(rtsp_url, options={
            'rtsp_transport': 'tcp',
            'max_delay': '5000000',
            'timeout': '10000000'
        })
        
        video_stream = container.streams.video[0]
        original_fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
        frame_interval = max(1, int(original_fps))
        
        frames_buffer = []
        alerts = []
        frame_count = 0
        clip_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        alert_container = st.empty()
        
        for frame in container.decode(video=0):
            frame_count += 1
            progress = min(clip_count / max_clips, 1.0)
            progress_bar.progress(progress)
            
            if frame_count % frame_interval == 0:
                img = frame.to_image()
                img = img.resize((224, 224))
                frames_buffer.append(img)
                
                if len(frames_buffer) >= 8:
                    clip_count += 1
                    status_text.text(f"🔍 Processing live stream clip {clip_count}/{max_clips}...")
                    
                    results = detect_fire_smoke(
                        frames_buffer[:8], processor, model, device, threshold
                    )
                    
                    if results and results['fire_detected']:
                        alert = {
                            'clip_number': clip_count,
                            'confidence': results['detection_confidence'],
                            'details': results
                        }
                        alerts.append(alert)
                        
                        # Show immediate alert
                        alert_container.markdown(
                            f"""<div class="alert-box">
                            🚨 <b>LIVE ALERT #{len(alerts)}</b><br>
                            Clip: {clip_count} | Confidence: {results['detection_confidence']:.3f}<br>
                            🔥 Fire: {results['fire_probability']:.3f} | 💨 Smoke: {results['smoke_probability']:.3f}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    
                    frames_buffer = frames_buffer[-2:]
                    
                    if clip_count >= max_clips:
                        break
        
        container.close()
        return alerts
        
    except Exception as e:
        st.error(f"RTSP connection error: {e}")
        return []

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔥 X-CLIP Fire & Smoke Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Model loading section
    if not st.session_state.model_loaded:
        st.markdown('<div class="info-box">📥 <b>First Time Setup:</b> Load X-CLIP model to begin detection</div>', unsafe_allow_html=True)
        
        if st.sidebar.button("🚀 Load X-CLIP Model", type="primary"):
            processor, model, device = load_xclip_model()
            if processor is not None:
                st.session_state.processor = processor
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("✅ X-CLIP model loaded successfully!")
                st.rerun()
    else:
        st.markdown('<div class="success-box">✅ <b>Model Ready:</b> X-CLIP loaded and ready for detection</div>', unsafe_allow_html=True)
        
        # Display system info
        st.sidebar.success("🤖 Model: X-CLIP Patch 32")
        st.sidebar.info(f"💻 Device: {st.session_state.device}")
        
        # Detection threshold
        threshold = st.sidebar.slider(
            "🎯 Detection Threshold", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.25, 
            step=0.05,
            help="Lower values = more sensitive detection"
        )
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📹 Video File Analysis", "🔴 RTSP Stream Test", "ℹ️ System Info"])
        
        with tab1:
            st.header("📹 Video File Fire & Smoke Detection")
            
            uploaded_file = st.file_uploader(
                "Choose a video file", 
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload CCTV footage or any video file for analysis"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.video(uploaded_file)
                    file_size = uploaded_file.size / (1024*1024)  # MB
                    st.caption(f"File size: {file_size:.1f} MB")
                
                with col1:
                    if st.button("🔍 Analyze Video for Fire/Smoke", type="primary"):
                        st.info("🎬 Processing video at 1 FPS using X-CLIP...")
                        
                        alerts, duration, clips_processed = process_video_file(
                            uploaded_file, 
                            st.session_state.processor,
                            st.session_state.model,
                            st.session_state.device,
                            threshold
                        )
                        
                        # Results
                        st.header("📊 Analysis Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Duration", format_video_duration(duration))
                        with col2:
                            st.metric("Clips Processed", clips_processed)
                        with col3:
                            st.metric("Alerts Generated", len(alerts))
                        with col4:
                            detection_rate = (len(alerts)/clips_processed*100) if clips_processed > 0 else 0
                            st.metric("Detection Rate", f"{detection_rate:.1f}%")
                        
                        if alerts:
                            st.markdown('<div class="alert-box">🚨 <b>FIRE/SMOKE DETECTED!</b></div>', unsafe_allow_html=True)
                            
                            # Detailed results
                            for i, alert in enumerate(alerts, 1):
                                with st.expander(f"🚨 Alert #{i} - [{alert['timestamp_formatted']}]"):
                                    details = alert['details']
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🔥 Fire Probability", f"{details['fire_probability']:.3f}")
                                    with col2:
                                        st.metric("💨 Smoke Probability", f"{details['smoke_probability']:.3f}")
                                    with col3:
                                        st.metric("📊 Confidence", f"{alert['confidence']:.3f}")
                                    
                                    st.json(details)
                            
                            # Export timestamps
                            timestamp_data = []
                            for alert in alerts:
                                timestamp_data.append({
                                    'video_time': alert['timestamp_formatted'],
                                    'confidence': alert['confidence']
                                })
                            
                            st.download_button(
                                label="📥 Download Timestamps (JSON)",
                                data=json.dumps(timestamp_data, indent=2),
                                file_name="fire_smoke_timestamps.json",
                                mime="application/json"
                            )
                        else:
                            st.markdown('<div class="success-box">✅ <b>No fire or smoke detected</b> - All clear!</div>', unsafe_allow_html=True)
        
        with tab2:
            st.header("🔴 Live RTSP Stream Testing")
            
            rtsp_url = st.text_input(
                "RTSP Stream URL",
                placeholder="rtsp://camera_ip:port/stream",
                help="Enter your camera's RTSP stream URL"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                max_clips = st.number_input("Max clips to test", min_value=1, max_value=50, value=10)
            with col2:
                timeout = st.number_input("Timeout (seconds)", min_value=10, max_value=300, value=60)
            
            if rtsp_url and st.button("🔴 Start RTSP Test", type="primary"):
                st.info("📡 Connecting to RTSP stream...")
                
                alerts = test_rtsp_stream(
                    rtsp_url,
                    st.session_state.processor,
                    st.session_state.model,
                    st.session_state.device,
                    max_clips,
                    threshold
                )
                
                st.header("📊 RTSP Test Results")
                
                if alerts:
                    st.markdown('<div class="alert-box">🚨 <b>LIVE ALERTS DETECTED!</b></div>', unsafe_allow_html=True)
                    
                    for alert in alerts:
                        details = alert['details']
                        st.warning(f"🚨 Alert from clip {alert['clip_number']} - Confidence: {alert['confidence']:.3f}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🔥 Fire", f"{details['fire_probability']:.3f}")
                        with col2:
                            st.metric("💨 Smoke", f"{details['smoke_probability']:.3f}")
                        with col3:
                            st.metric("📊 Combined", f"{details['fire_smoke_combined']:.3f}")
                else:
                    st.markdown('<div class="success-box">✅ <b>No alerts</b> from RTSP stream test</div>', unsafe_allow_html=True)
        
        with tab3:
            st.header("ℹ️ System Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 Model Details")
                st.info("""
                **Model**: microsoft/xclip-base-patch32  
                **Type**: Video-Text Understanding  
                **Framework**: PyTorch + Transformers  
                **Input**: 8-frame video clips  
                **Processing**: 1 FPS sampling
                """)
                
                st.subheader("🎯 Detection Features")
                st.success("""
                ✅ Real fire and flame detection  
                ✅ Smoke detection (all types)  
                ✅ Reduced false positives  
                ✅ Video timeline analysis  
                ✅ Live RTSP stream support  
                ✅ Confidence scoring
                """)
            
            with col2:
                st.subheader("💻 System Status")
                system_info = {
                    "CUDA Available": torch.cuda.is_available(),
                    "Device": st.session_state.device,
                    "Model Loaded": st.session_state.model_loaded,
                    "Detection Threshold": threshold,
                }
                
                for key, value in system_info.items():
                    if isinstance(value, bool):
                        st.metric(key, "✅ Yes" if value else "❌ No")
                    else:
                        st.metric(key, str(value))
                
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    st.metric("GPU Memory", f"{gpu_memory:.1f} GB")
                
                st.subheader("📋 Usage Instructions")
                st.markdown("""
                1. **Load Model** (first time only)
                2. **Upload video** or **enter RTSP URL**
                3. **Adjust threshold** if needed
                4. **Run detection** and review results
                5. **Download timestamps** for incidents
                """)

if __name__ == "__main__":
    main()
