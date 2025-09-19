# streamlit_fire_detection.py - Updated with Enhanced RTSP Error Handling
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
import socket
import subprocess
import platform
import re
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
    .warning-box {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        background-color: #fffdf3;
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

# Enhanced RTSP diagnostic functions
def check_network_connectivity(ip_address, port=554):
    """Check if camera IP is reachable"""
    try:
        socket.setdefaulttimeout(5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((ip_address, port))
        sock.close()
        return result == 0
    except Exception as e:
        return False

def ping_camera(ip_address):
    """Ping camera IP to check basic connectivity"""
    try:
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '1', ip_address]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False

def extract_ip_from_rtsp(rtsp_url):
    """Extract IP address from RTSP URL"""
    try:
        ip_match = re.search(r'://(?:[^:]+:[^@]+@)?(\d+\.\d+\.\d+\.\d+)', rtsp_url)
        return ip_match.group(1) if ip_match else None
    except:
        return None

# Model loading function
@st.cache_resource
def load_xclip_model():
    """Load X-CLIP model with caching"""
    try:
        with st.spinner("🔄 Loading X-CLIP model for first time..."):
            from transformers import AutoProcessor, AutoModel
            
            processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
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

# Enhanced RTSP testing function
def test_rtsp_connection_enhanced(rtsp_url, processor, model, device, max_clips=3, threshold=0.25):
    """Enhanced RTSP testing with comprehensive error handling"""
    
    # Extract IP from RTSP URL
    camera_ip = extract_ip_from_rtsp(rtsp_url)
    
    st.markdown('<div class="info-box">🔍 <b>Diagnosing RTSP Connection...</b></div>', unsafe_allow_html=True)
    
    # Diagnostic results container
    diag_container = st.container()
    
    with diag_container:
        if camera_ip:
            st.write(f"📍 **Camera IP**: `{camera_ip}`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("🌐 **Network Tests:**")
                # Ping test
                with st.spinner("Testing ping..."):
                    ping_status = ping_camera(camera_ip)
                if ping_status:
                    st.success("✅ Ping successful")
                else:
                    st.error("❌ Ping failed")
            
            with col2:
                st.write("🔌 **Port Tests:**")
                # Port connectivity test
                with st.spinner("Testing port 554..."):
                    port_status = check_network_connectivity(camera_ip, 554)
                if port_status:
                    st.success("✅ Port 554 accessible")
                else:
                    st.error("❌ Port 554 blocked")
                    
                # Test alternative port
                with st.spinner("Testing port 8554..."):
                    alt_port_status = check_network_connectivity(camera_ip, 8554)
                if alt_port_status:
                    st.success("✅ Port 8554 accessible")
                else:
                    st.warning("⚠️ Port 8554 not accessible")
    
    # Generate RTSP URL variations
    rtsp_variations = [
        rtsp_url,  # Original
        rtsp_url.replace(':554/', ':8554/'),  # Alternative port
        rtsp_url.replace('Streaming/Channels/1', 'stream1'),
        rtsp_url.replace('Streaming/Channels/1', 'live'),
        rtsp_url.replace('Streaming/Channels/1', 'h264/ch1/main/av_stream'),
        rtsp_url.replace('Streaming/Channels/1', 'cam/realmonitor?channel=1&subtype=0'),
    ]
    
    connection_options = [
        {
            'rtsp_transport': 'tcp',
            'rtsp_flags': 'prefer_tcp',
            'timeout': '5000000',
            'max_delay': '2000000',
            'buffer_size': '1024000'
        },
        {
            'rtsp_transport': 'udp',
            'timeout': '8000000',
            'max_delay': '3000000',
        },
        {
            'rtsp_transport': 'tcp',
            'timeout': '10000000',
            'max_delay': '5000000',
            'rtsp_flags': 'prefer_tcp+listen'
        }
    ]
    
    st.markdown('<div class="info-box">🔄 <b>Testing Connection Methods...</b></div>', unsafe_allow_html=True)
    
    # Progress tracking
    total_attempts = min(len(rtsp_variations), 4) * len(connection_options)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    attempt = 0
    
    for i, url_variant in enumerate(rtsp_variations[:4]):  # Test first 4 variations
        for j, options in enumerate(connection_options):
            attempt += 1
            progress_bar.progress(attempt / total_attempts)
            
            try:
                status_text.text(f"🔗 Attempt {attempt}/{total_attempts}: Testing connection...")
                
                # Show current attempt details
                with st.expander(f"Attempt {attempt} Details", expanded=False):
                    st.code(f"URL: {url_variant}")
                    st.json(options)
                
                # Try connection
                container = av.open(url_variant, options=options)
                video_stream = container.streams.video[0]
                
                st.success(f"✅ **Connection successful on attempt {attempt}!**")
                st.info(f"📹 Stream: {video_stream.width}x{video_stream.height}")
                st.info(f"🔗 Working URL: `{url_variant}`")
                
                # Test frame processing
                alerts = test_rtsp_frames(container, processor, model, device, max_clips, threshold)
                
                status_text.text("✅ RTSP connection and processing successful!")
                progress_bar.progress(1.0)
                
                return alerts, url_variant
                
            except Exception as e:
                error_msg = str(e)
                
                # Categorize errors
                if "Connection timed out" in error_msg:
                    error_type = "⏰ Timeout"
                    color = "warning"
                elif "Connection refused" in error_msg:
                    error_type = "🚫 Connection Refused"
                    color = "error"
                elif "No route to host" in error_msg:
                    error_type = "🌐 Network Unreachable"
                    color = "error"
                elif "Authentication" in error_msg or "401" in error_msg:
                    error_type = "🔐 Authentication Failed"
                    color = "error"
                else:
                    error_type = "❌ Other Error"
                    color = "warning"
                
                # Show error in expandable section
                with st.expander(f"Attempt {attempt}: {error_type}", expanded=False):
                    if color == "error":
                        st.error(f"{error_type}: {error_msg[:200]}...")
                    else:
                        st.warning(f"{error_type}: {error_msg[:200]}...")
                
                continue
    
    # All attempts failed
    progress_bar.progress(1.0)
    status_text.text("❌ All connection attempts failed")
    
    show_rtsp_troubleshooting_guide(camera_ip, rtsp_url)
    return [], None

def test_rtsp_frames(container, processor, model, device, max_clips, threshold):
    """Process frames from successful RTSP connection"""
    try:
        video_stream = container.streams.video[0]
        original_fps = float(video_stream.average_rate) if video_stream.average_rate else 25.0
        frame_interval = max(1, int(original_fps // 2))
        
        frames_buffer = []
        alerts = []
        frame_count = 0
        clip_count = 0
        
        frame_progress = st.progress(0)
        frame_status = st.empty()
        
        st.success("🎬 Processing frames from RTSP stream...")
        
        for frame in container.decode(video=0):
            frame_count += 1
            progress = min(clip_count / max_clips, 1.0)
            frame_progress.progress(progress)
            
            if frame_count % frame_interval == 0:
                try:
                    img = frame.to_image()
                    img = img.resize((224, 224))
                    frames_buffer.append(img)
                    
                    if len(frames_buffer) >= 8:
                        clip_count += 1
                        frame_status.text(f"🔍 Analyzing clip {clip_count}/{max_clips}...")
                        
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
                            
                            st.markdown(
                                f'''<div class="alert-box">
                                🚨 <b>LIVE ALERT #{len(alerts)}</b><br>
                                Clip {clip_count} | Confidence: {results['detection_confidence']:.3f}<br>
                                🔥 Fire: {results['fire_probability']:.3f} | 💨 Smoke: {results['smoke_probability']:.3f}
                                </div>''', 
                                unsafe_allow_html=True
                            )
                        else:
                            st.info(f"✅ Clip {clip_count}: Normal scene")
                        
                        frames_buffer = frames_buffer[-2:]
                        
                        if clip_count >= max_clips:
                            break
                            
                except Exception as frame_error:
                    st.warning(f"⚠️ Frame {frame_count} error: {str(frame_error)[:100]}")
                    continue
        
        container.close()
        frame_status.text(f"✅ Processing complete - {clip_count} clips analyzed")
        frame_progress.progress(1.0)
        
        return alerts
        
    except Exception as e:
        st.error(f"Frame processing error: {e}")
        if 'container' in locals():
            container.close()
        return []

def show_rtsp_troubleshooting_guide(camera_ip, original_url):
    """Show comprehensive troubleshooting guide"""
    
    st.markdown('<div class="alert-box">❌ <b>All RTSP connection attempts failed</b></div>', unsafe_allow_html=True)
    
    with st.expander("🔧 **Comprehensive Troubleshooting Guide**", expanded=True):
        
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Quick Fixes", "🌐 Network", "📹 Camera", "🔗 URLs"])
        
        with tab1:
            st.markdown("### 🚀 **Try These URLs First:**")
            
            quick_urls = [
                original_url.replace(':554/', ':8554/'),
                original_url.replace('Streaming/Channels/1', 'stream1'),
                original_url.replace('Streaming/Channels/1', 'live'),
                original_url.replace('Streaming/Channels/1', 'h264/ch1/main/av_stream'),
                f"rtsp://admin:OSUUTH@{camera_ip}:554/cam/realmonitor?channel=1&subtype=0"
            ]
            
            for i, url in enumerate(quick_urls, 1):
                st.code(f"{i}. {url}")
            
            st.markdown("### 🎯 **Immediate Actions:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🌐 Test Web Interface"):
                    st.info(f"Open in browser: http://{camera_ip}")
                    st.caption("If web interface works, RTSP should be available")
            
            with col2:
                if st.button("📱 Generate VLC Command"):
                    st.code(f"vlc {original_url}")
                    st.caption("Test URL with VLC Media Player")
        
        with tab2:
            st.markdown("### 🌐 **Network Troubleshooting:**")
            
            if camera_ip:
                st.markdown(f"""
                **Manual Network Tests:**
                ```
                # Ping test
                ping {camera_ip}
                
                # Port test (Windows)
                telnet {camera_ip} 554
                
                # Port test (Linux/Mac)
                nc -zv {camera_ip} 554
                ```
                """)
            
            st.markdown("""
            **Common Network Issues:**
            - ❌ Camera and computer on different subnets
            - ❌ Firewall blocking port 554/8554
            - ❌ Router blocking RTSP traffic
            - ❌ Network congestion or packet loss
            - ❌ VPN interfering with local network
            """)
        
        with tab3:
            st.markdown("### 📹 **Camera Configuration:**")
            
            st.markdown("""
            **Check Camera Settings:**
            1. **RTSP Service**: Must be enabled
            2. **User Permissions**: RTSP access allowed
            3. **Connection Limit**: May have max concurrent connections
            4. **Stream Settings**: Main/sub stream configuration
            5. **Authentication**: Username/password correct
            
            **Common Camera Issues:**
            - ❌ RTSP disabled in camera settings
            - ❌ User lacks RTSP permissions  
            - ❌ Maximum connections reached
            - ❌ Wrong credentials or expired password
            - ❌ Camera firmware needs update
            """)
            
            st.info("💡 **Tip**: Access camera web interface to verify RTSP settings")
        
        with tab4:
            st.markdown("### 🔗 **Brand-Specific URL Formats:**")
            
            brand_formats = {
                "Hikvision": [
                    "/Streaming/Channels/1",
                    "/Streaming/Channels/101", 
                    "/h264/ch1/main/av_stream"
                ],
                "Dahua": [
                    "/cam/realmonitor?channel=1&subtype=0",
                    "/live/ch1"
                ],
                "Axis": [
                    "/axis-media/media.amp",
                    "/mjpg/video.mjpg"
                ],
                "Foscam": [
                    "/videoMain",
                    "/video.cgi"
                ],
                "Generic": [
                    "/stream1",
                    "/live", 
                    "/video",
                    "/cam1"
                ]
            }
            
            for brand, formats in brand_formats.items():
                with st.expander(f"📹 {brand} URLs"):
                    for fmt in formats:
                        if camera_ip:
                            st.code(f"rtsp://admin:OSUUTH@{camera_ip}:554{fmt}")

# Video processing function (unchanged)
def process_video_file(video_file, processor, model, device, threshold=0.25):
    """Process uploaded video file"""
    try:
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
        os.remove("temp_video.mp4")
        
        return alerts, total_duration, detection_count
        
    except Exception as e:
        st.error(f"Video processing error: {e}")
        return [], 0, 0

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
        tab1, tab2, tab3 = st.tabs(["📹 Video File Analysis", "🔴 Enhanced RTSP Test", "ℹ️ System Info"])
        
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
                    file_size = uploaded_file.size / (1024*1024)
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
                        
                        # Results display (same as before)
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
                            
                            # Export functionality
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
            st.header("🔴 Enhanced RTSP Stream Testing")
            st.caption("Advanced diagnostics and multiple connection methods")
            
            # Show the problematic URL as an example
            st.markdown('<div class="warning-box">⚠️ <b>Current Issue:</b> Connection timeout to camera</div>', unsafe_allow_html=True)
            
            rtsp_url = st.text_input(
                "RTSP Stream URL",
                value="rtsp://admin:OSUUTH@192.168.10.211:554/Streaming/Channels/1",
                help="Your camera's RTSP stream URL with credentials"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                max_clips = st.number_input("Max test clips", min_value=1, max_value=10, value=3)
            with col2:
                run_diagnostics = st.checkbox("Run network diagnostics", value=True)
            with col3:
                show_attempts = st.checkbox("Show all attempts", value=False)
            
            if rtsp_url and st.button("🔴 Start Enhanced RTSP Test", type="primary"):
                alerts, working_url = test_rtsp_connection_enhanced(
                    rtsp_url,
                    st.session_state.processor,
                    st.session_state.model,
                    st.session_state.device,
                    max_clips,
                    threshold
                )
                
                st.header("📊 Enhanced RTSP Test Results")
                
                if working_url:
                    st.markdown(f'<div class="success-box">✅ <b>Connection Success!</b><br>Working URL: <code>{working_url}</code></div>', unsafe_allow_html=True)
                
                if alerts:
                    st.markdown('<div class="alert-box">🚨 <b>LIVE FIRE/SMOKE ALERTS DETECTED!</b></div>', unsafe_allow_html=True)
                    
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
                    if working_url:
                        st.markdown('<div class="success-box">✅ <b>No fire/smoke detected</b> in RTSP test</div>', unsafe_allow_html=True)
        
        with tab3:
            st.header("ℹ️ System Information & Help")
            
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
                ✅ Enhanced RTSP diagnostics  
                ✅ Multiple connection methods  
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
                
                st.subheader("🔧 RTSP Troubleshooting")
                st.markdown("""
                **Common Solutions:**
                1. Try port 8554 instead of 554
                2. Use simple paths like `/stream1`
                3. Check camera web interface
                4. Verify network connectivity
                5. Test with VLC first
                """)

if __name__ == "__main__":
    main()