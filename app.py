"""
Chicken Detection Web App
A Streamlit application for detecting and counting chickens in images
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import tempfile
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import gdown

# Page configuration
st.set_page_config(
    page_title="🐔 Chicken Detection System ",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🐔 Chicken Detection System شركة الدجاج شركة واااااقفة</h1>', unsafe_allow_html=True)

st.markdown(" <p style="text-align: center;">  ### Rayen Hamdaoui -- Hedi Nemer </p>")
st.markdown("### Hakim Moahemmed Aziz -- Meddeb Youssef")

st.markdown("### Upload images to detect and count chickens automatically")

# ============================================================================
# MODEL LOADING (Cached)
# ============================================================================

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model (cached to avoid reloading)"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def download_and_load_model():
    """Download model from Google Drive and load it"""
    model_path = 'models/best.pt'
    
    # Always check if file is Git LFS pointer
    needs_download = False
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        
        # If file is suspiciously small (< 1MB), it's likely a Git LFS pointer
        if file_size < 1024 * 1024:  # Less than 1MB
            st.warning(f"⚠️ Model file is only {file_size} bytes. Downloading actual model from Google Drive...")
            os.remove(model_path)
            needs_download = True
    else:
        needs_download = True
    
    if needs_download:
        os.makedirs('models', exist_ok=True)
        
        # YOUR GOOGLE DRIVE FILE ID - REPLACE THIS!
        file_id = '1VBnwiGuzU-Q6yjiwWHLj_0QBVVmkBX_A'
        url = f'https://drive.google.com/uc?id={file_id}'
        
        try:
            with st.spinner('📥 Downloading model from Google Drive... (this will take a minute)'):
                gdown.download(url, model_path, quiet=False)
            
            # Verify download
            new_size = os.path.getsize(model_path)
            st.success(f'✅ Model downloaded successfully! Size: {new_size / (1024*1024):.2f} MB')
        except Exception as e:
            st.error(f'❌ Failed to download model: {e}')
            st.error('Please check your Google Drive file ID and sharing settings.')
            st.stop()
    
    # Load model
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f'❌ Failed to load model: {e}')
        st.stop()

model = download_and_load_model()

if model is None:
    st.error("⚠️ Failed to load model. Please check the model path.")
    st.stop()











# ============================================================================
# SIDEBAR - Settings and Controls
# ============================================================================

st.sidebar.header("⚙️ Detection Settings")

# Confidence threshold slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Adjust detection sensitivity. Lower = more detections (but more false positives)"
)

st.sidebar.markdown(f"**Current threshold:** `{conf_threshold:.2f}`")

# Display options
st.sidebar.header("📊 Display Options")
show_confidence = st.sidebar.checkbox("Show confidence scores", value=True)
show_bounding_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
show_labels = st.sidebar.checkbox("Show labels", value=True)

# Download options
st.sidebar.header("💾 Export Options")
enable_csv_export = st.sidebar.checkbox("Enable CSV export", value=True)

st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload one or more images
2. Adjust confidence threshold
3. View detection results
4. Download results if needed
""")

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def detect_chickens(image, conf_threshold):
    """
    Run chicken detection on an image
    
    Args:
        image: PIL Image
        conf_threshold: Confidence threshold
    
    Returns:
        results object from YOLO
    """
    results = model(image, conf=conf_threshold, verbose=False)
    return results[0]

def draw_detections(image, result, show_conf=True):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: PIL Image
        result: YOLO result object
        show_conf: Whether to show confidence scores
    
    Returns:
        Annotated PIL Image
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Get boxes
    boxes = result.boxes
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf[0].cpu().numpy()
        
        # Draw bounding box
        if show_bounding_boxes:
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label
        if show_labels:
            label = f"Chicken"
            if show_conf:
                label += f" {conf:.2%}"
            
            # Get text size for background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw background rectangle
            cv2.rectangle(img_array, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(img_array, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return Image.fromarray(img_array)

def get_detection_stats(result):
    """
    Extract statistics from detection results
    
    Returns:
        Dictionary with detection statistics
    """
    boxes = result.boxes
    count = len(boxes)
    
    if count > 0:
        confidences = [box.conf[0].cpu().numpy() for box in boxes]
        avg_conf = np.mean(confidences)
        max_conf = np.max(confidences)
        min_conf = np.min(confidences)
        
        # Get bounding box sizes
        sizes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            sizes.append(width * height)
        
        avg_size = np.mean(sizes)
    else:
        avg_conf = max_conf = min_conf = avg_size = 0.0
    
    return {
        'count': count,
        'avg_confidence': avg_conf,
        'max_confidence': max_conf,
        'min_confidence': min_conf,
        'avg_size': avg_size
    }

# ============================================================================
# FILE UPLOAD
# ============================================================================

uploaded_files = st.file_uploader(
    "Upload Images (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="You can upload multiple images at once"
)

if uploaded_files:
    st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Results", "📊 Statistics", "📥 Export"])
    
    # Store all results
    all_results = []
    
    with tab1:
        st.subheader("Detection Results")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Read image
            image = Image.open(uploaded_file)
            
            # Create columns for original and detected images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Original Image:** {uploaded_file.name}")
                st.image(image, use_container_width=True)
            
            # Run detection
            with st.spinner(f"Detecting chickens in {uploaded_file.name}..."):
                result = detect_chickens(image, conf_threshold)
                stats = get_detection_stats(result)
            
            # Draw detections
            annotated_image = draw_detections(
                image, 
                result, 
                show_conf=show_confidence
            )
            
            with col2:
                st.markdown(f"**Detected Image:** {stats['count']} chicken(s) found")
                st.image(annotated_image, use_container_width=True)
            
            # Display metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("🐔 Chickens", stats['count'])
            with metric_cols[1]:
                st.metric("📊 Avg Confidence", f"{stats['avg_confidence']:.1%}")
            with metric_cols[2]:
                st.metric("📈 Max Confidence", f"{stats['max_confidence']:.1%}")
            with metric_cols[3]:
                st.metric("📉 Min Confidence", f"{stats['min_confidence']:.1%}" if stats['count'] > 0 else "N/A")
            
            # Store results
            all_results.append({
                'filename': uploaded_file.name,
                'chicken_count': stats['count'],
                'avg_confidence': stats['avg_confidence'],
                'max_confidence': stats['max_confidence'],
                'min_confidence': stats['min_confidence'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Add download button for individual image
            buf = io.BytesIO()
            annotated_image.save(buf, format='PNG')
            st.download_button(
                label=f"⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"detected_{uploaded_file.name}",
                mime="image/png",
                key=f"download_{idx}"
            )
            
            st.markdown("---")
    
    with tab2:
        st.subheader("📊 Detection Statistics")
        
        if all_results:
            # Overall statistics
            total_chickens = sum([r['chicken_count'] for r in all_results])
            avg_chickens = total_chickens / len(all_results)
            images_with_chickens = sum([1 for r in all_results if r['chicken_count'] > 0])
            
            # Display overall metrics
            st.markdown("### Overall Summary")
            overview_cols = st.columns(4)
            with overview_cols[0]:
                st.metric("📁 Total Images", len(all_results))
            with overview_cols[1]:
                st.metric("🐔 Total Chickens", total_chickens)
            with overview_cols[2]:
                st.metric("📊 Avg per Image", f"{avg_chickens:.2f}")
            with overview_cols[3]:
                st.metric("✅ Detection Rate", f"{images_with_chickens/len(all_results)*100:.1f}%")
            
            st.markdown("---")
            
            # Create visualizations
            st.markdown("### Visual Analytics")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Bar chart of chicken counts
                df = pd.DataFrame(all_results)
                st.bar_chart(df.set_index('filename')['chicken_count'])
                st.caption("Chicken count per image")
            
            with chart_col2:
                # Confidence distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                confidences = [r['avg_confidence'] for r in all_results if r['chicken_count'] > 0]
                if confidences:
                    ax.hist(confidences, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
                    ax.set_xlabel('Average Confidence')
                    ax.set_ylabel('Number of Images')
                    ax.set_title('Confidence Score Distribution')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.info("No detections to show confidence distribution")
            
            # Detailed table
            st.markdown("### Detailed Results")
            df_display = pd.DataFrame(all_results)
            df_display['avg_confidence'] = df_display['avg_confidence'].apply(lambda x: f"{x:.2%}")
            df_display['max_confidence'] = df_display['max_confidence'].apply(lambda x: f"{x:.2%}")
            df_display['min_confidence'] = df_display['min_confidence'].apply(lambda x: f"{x:.2%}" if x > 0 else "N/A")
            st.dataframe(df_display, use_container_width=True)
    
    with tab3:
        st.subheader("💾 Export Results")
        
        if enable_csv_export and all_results:
            # Convert to DataFrame
            df = pd.DataFrame(all_results)
            
            # Convert to CSV
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Report",
                data=csv,
                file_name=f"chicken_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ CSV report ready for download!")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Enable CSV export in the sidebar to download results")

else:
    # Display instructions when no files uploaded
    st.info("👆 Upload images using the file uploader above to start detecting chickens!")
    
    st.markdown("""
    ### Features:
    - 🎯 Adjustable confidence threshold
    - 🖼️ Process single or multiple images
    - 📊 Real-time statistics and analytics
    - 💾 Export results as CSV
    - ⬇️ Download annotated images
    
    ### Tips:
    - **Lower threshold (0.1-0.2):** Detects more chickens but may include false positives
    - **Medium threshold (0.25-0.4):** Balanced detection (recommended)
    - **Higher threshold (0.5+):** Only very confident detections, may miss some chickens
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Chicken Detection System v1.0 | Powered by YOLOv8 🐔</p>
    </div>
""", unsafe_allow_html=True)












