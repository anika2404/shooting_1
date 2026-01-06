import streamlit as st
import cv2
import math
import statistics
import numpy as np
from PIL import Image
import io

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Shooting Analysis System",
    page_icon="🎯",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff6b6b !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #4ecdc4 !important;
        font-size: 2.5em !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 2px dashed rgba(255, 107, 107, 0.5);
        padding: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: bold;
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.4);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(78, 205, 196, 0.6);
        transform: translateY(-3px);
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }
    
    /* Text color */
    p, label, .stMarkdown {
        color: white !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 107, 107, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# ERROR RULES DATABASE
# =====================================================
ERROR_RULES = {
    "mechanical": {
        "right": {
            "error": "Trigger finger pushing right",
            "cause": "Too much finger on trigger / sideways pull",
            "correction": "Use center of first finger pad, pull straight back"
        },
        "left": {
            "error": "Trigger finger pulling left",
            "cause": "Trigger finger too shallow or finger tension",
            "correction": "Increase finger contact, relax trigger finger"
        },
        "low": {
            "error": "Anticipation / wrist dropping",
            "cause": "Recoil anticipation or grip relaxation",
            "correction": "Improve follow-through, keep wrist locked"
        },
        "high": {
            "error": "Heeling / pushing up",
            "cause": "Palm pressure during trigger break",
            "correction": "Reduce palm pressure, neutral grip"
        }
    },
    "physical": {
        "wide_group": {
            "error": "Hold instability or fatigue",
            "cause": "Muscle fatigue or weak stance",
            "correction": "Improve core stability, reduce hold time"
        },
        "vertical_spread": {
            "error": "Breathing inconsistency",
            "cause": "Shot fired outside natural respiratory pause",
            "correction": "Fire during natural pause"
        }
    },
    "mental": {
        "outlier": {
            "error": "Loss of focus / routine break",
            "cause": "Distraction or rushed shot",
            "correction": "Reset routine before every shot"
        }
    }
}

THRESHOLDS = {
    "shift_mm": 1.5,
    "group_mm": 7.0,
    "vertical_ratio": 1.4
}

# =====================================================
# ANALYSIS FUNCTION
# =====================================================
def analyze_target(image_file):
    """Analyze uploaded target image"""
    
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "❌ Could not read image"
    
    # Image preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Shot detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shots_pixel = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 300:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                shots_pixel.append((cx, cy))
    
    if len(shots_pixel) < 10:
        return None, f"❌ Only {len(shots_pixel)} shots detected. Need at least 10."
    
    # Convert to target coordinates
    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h // 2
    SCALE = 0.1
    
    shots = []
    for x, y in shots_pixel:
        rx = (x - center_x) * SCALE
        ry = (center_y - y) * SCALE
        shots.append((rx, ry))
    
    # Calculate temp center for sorting
    mean_x_temp = statistics.mean([x for x, y in shots])
    mean_y_temp = statistics.mean([y for x, y in shots])
    
    shot_with_dist = []
    for x, y in shots:
        d = math.sqrt((x - mean_x_temp)**2 + (y - mean_y_temp)**2)
        shot_with_dist.append((x, y, d))
    
    # Select 10 closest shots
    shot_with_dist.sort(key=lambda s: s[2])
    shot_with_dist = shot_with_dist[:10]
    
    sequenced_shots = []
    for i, (x, y, d) in enumerate(shot_with_dist, start=1):
        sequenced_shots.append({
            "name": f"Shot {i}",
            "x": x,
            "y": y,
            "distance": d
        })
    
    # Statistical analysis
    x_vals = [s["x"] for s in sequenced_shots]
    y_vals = [s["y"] for s in sequenced_shots]
    
    mean_x = statistics.mean(x_vals)
    mean_y = statistics.mean(y_vals)
    
    distances = [
        math.sqrt((x - mean_x)**2 + (y - mean_y)**2)
        for x, y in zip(x_vals, y_vals)
    ]
    
    group_size = max(distances)
    
    # Error classification
    mechanical = []
    physical = []
    mental = []
    
    if mean_x > THRESHOLDS["shift_mm"]:
        mechanical.append(ERROR_RULES["mechanical"]["right"])
    elif mean_x < -THRESHOLDS["shift_mm"]:
        mechanical.append(ERROR_RULES["mechanical"]["left"])
    
    if mean_y < -THRESHOLDS["shift_mm"]:
        mechanical.append(ERROR_RULES["mechanical"]["low"])
    elif mean_y > THRESHOLDS["shift_mm"]:
        mechanical.append(ERROR_RULES["mechanical"]["high"])
    
    if group_size > THRESHOLDS["group_mm"]:
        physical.append(ERROR_RULES["physical"]["wide_group"])
    
    if statistics.stdev(y_vals) > statistics.stdev(x_vals) * THRESHOLDS["vertical_ratio"]:
        physical.append(ERROR_RULES["physical"]["vertical_spread"])
    
    avg_d = statistics.mean(distances)
    std_d = statistics.stdev(distances)
    
    if any(d > avg_d + std_d for d in distances):
        mental.append(ERROR_RULES["mental"]["outlier"])
    
    # Draw circles on image
    img_with_circles = img.copy()
    for (x, y) in shots_pixel[:10]:
        cv2.circle(img_with_circles, (x, y), 6, (0, 0, 255), 2)
    
    return {
        "shots": sequenced_shots,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "group_size": group_size,
        "mechanical": mechanical,
        "physical": physical,
        "mental": mental,
        "image": img_with_circles
    }, None

# =====================================================
# STREAMLIT APP
# =====================================================

# Header
st.title("🎯 Shooting Analysis System")
st.caption("AI-Powered 10-Shot Grouping & Error Detection")
st.divider()

# File upload section
st.subheader("📸 Upload Target Image")
uploaded_file = st.file_uploader(
    "Choose an image of your target with pellet holes",
    type=['png', 'jpg', 'jpeg', 'bmp']
)

if uploaded_file is not None:
    # Show preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Target", use_container_width=True)
    
    with col2:
        st.info("📋 **File Info**")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        st.write(f"**Type:** {uploaded_file.type}")
    
    st.divider()
    
    # Analysis button
    if st.button("🔍 Analyze Shots", use_container_width=True):
        with st.spinner("🔄 Analyzing target image..."):
            # Reset file pointer
            uploaded_file.seek(0)
            result, error = analyze_target(uploaded_file)
            
            if error:
                st.error(error)
            else:
                st.success("✅ Analysis Complete!")
                st.divider()
                
                # =====================================================
                # RESULTS DASHBOARD
                # =====================================================
                
                st.subheader("📊 Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Group Center",
                        f"({result['mean_x']:.2f}, {result['mean_y']:.2f})",
                        help="X, Y coordinates in millimeters"
                    )
                
                with col2:
                    st.metric(
                        "📏 Group Size",
                        f"{result['group_size']:.2f} mm",
                        help="Maximum spread of shots"
                    )
                
                with col3:
                    precision_score = max(0, min(100, 100 - result['group_size'] * 10))
                    st.metric(
                        "📊 Precision Score",
                        f"{precision_score:.0f}/100",
                        help="Higher is better"
                    )
                
                with col4:
                    st.metric(
                        "✅ Shots Analyzed",
                        "10",
                        help="Total shots detected and analyzed"
                    )
                
                st.divider()
                
                # Shot visualization and data
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.subheader("🎯 Detected Shots")
                    # Convert BGR to RGB for display
                    img_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, use_container_width=True)
                
                with col2:
                    st.subheader("📋 Shot Data")
                    for shot in result['shots']:
                        with st.container():
                            st.markdown(f"""
                            **{shot['name']}**  
                            X: `{shot['x']:.2f} mm` | Y: `{shot['y']:.2f} mm`
                            """)
                            st.markdown("---")
                
                st.divider()
                
                # =====================================================
                # ERROR ANALYSIS
                # =====================================================
                
                st.subheader("⚠️ Error Analysis & Corrections")
                
                total_errors = (len(result['mechanical']) + 
                               len(result['physical']) + 
                               len(result['mental']))
                
                if total_errors == 0:
                    st.success("### ✅ Excellent! No significant errors detected.")
                    st.balloons()
                else:
                    col1, col2, col3 = st.columns(3)
                    
                    # Mechanical Errors
                    with col1:
                        st.markdown("### 🔧 Mechanical Errors")
                        if not result['mechanical']:
                            st.info("None detected ✓")
                        else:
                            for err in result['mechanical']:
                                with st.expander(f"⚠️ {err['error']}", expanded=True):
                                    st.warning(f"**Cause:** {err['cause']}")
                                    st.success(f"**Correction:** {err['correction']}")
                    
                    # Physical Errors
                    with col2:
                        st.markdown("### 💪 Physical Errors")
                        if not result['physical']:
                            st.info("None detected ✓")
                        else:
                            for err in result['physical']:
                                with st.expander(f"⚠️ {err['error']}", expanded=True):
                                    st.warning(f"**Cause:** {err['cause']}")
                                    st.success(f"**Correction:** {err['correction']}")
                    
                    # Mental Errors
                    with col3:
                        st.markdown("### 🧠 Mental Errors")
                        if not result['mental']:
                            st.info("None detected ✓")
                        else:
                            for err in result['mental']:
                                with st.expander(f"⚠️ {err['error']}", expanded=True):
                                    st.warning(f"**Cause:** {err['cause']}")
                                    st.success(f"**Correction:** {err['correction']}")

else:
    # Instructions when no file uploaded
    st.info("👆 Please upload a target image to begin analysis")
    
    with st.expander("📖 How to use this system"):
        st.markdown("""
        1. **Take a clear photo** of your shooting target
        2. **Upload the image** using the file uploader above
        3. **Click 'Analyze Shots'** to process the image
        4. **Review the results** including:
           - Shot grouping statistics
           - Individual shot coordinates
           - Detected errors and corrections
        
        **Requirements:**
        - Image should show at least 10 pellet holes
        - Target should be clearly visible
        - Good lighting and contrast
        """)

# Footer
st.divider()
st.caption("🎯 Shooting Analysis System | Powered by Computer Vision & AI")