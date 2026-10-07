import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from collections import OrderedDict
import matplotlib.pyplot as plt

from image_effects import (
    apply_grayscale, apply_invert, apply_brightness, apply_contrast,
    apply_histogram_stretching, apply_histogram_equalization,
    apply_salt_and_pepper, apply_gaussian_noise,
    apply_rotate, apply_flip_horizontal, apply_flip_vertical,
    apply_blur, apply_sharpen, apply_threshold, apply_otsu_threshold,
    apply_canny, apply_roberts, apply_prewitt, apply_sobel,
    apply_dilate, apply_erode, apply_open, apply_close,
    apply_kmeans, apply_fcm, apply_pcm,
    compute_mse, compute_psnr,apply_periodic_noise,apply_poisson_noise,apply_resize,apply_laplacian_of_gaussian,apply_laplacian
)
from styles import CSS_STYLES

st.set_page_config(
    page_title="Image Processing Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(CSS_STYLES, unsafe_allow_html=True)

#dict to keep track of active effects
if 'effects_pipeline' not in st.session_state:
    st.session_state.effects_pipeline = OrderedDict()
# set to keep track of active effects that have been disabled
if 'effects_disabled' not in st.session_state:
    st.session_state.effects_disabled = set()
# image variables
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
#active effects counter
if 'effect_counter' not in st.session_state:
    st.session_state.effect_counter = 0


# better display names for parameters instead of the variable names
PARAM_DISPLAY_NAMES = {
    "recolor_mode": "Class Colors",
    "kernel_size": "Kernel Size",
    "kernel": "Kernel",
    "angle": "Angle",
    "amount": "Amount",
    "var": "Variance",
    "value": "Value",
    "alpha": "Alpha",
    "thresh": "Threshold",
    "iterations": "Iterations",
    "k": "K (Clusters)",
    "m": "Fuzziness (m)",
}


# functions and their params list, seperated by category
EFFECTS_CATEGORIES = {
    "🎨 Color": {
        "Brightness": {"func": apply_brightness, "params": {"value": (-100, 100, 5)}},
        "Contrast": {"func": apply_contrast, "params": {"alpha": (0.5, 3.0, 0.1)}},
        "Grayscale": {"func": apply_grayscale, "params": {}},
        "Black/White": {"func": apply_otsu_threshold, "params": {}},
        
        "Invert Colors": {"func": apply_invert, "params": {}},
        "Histogram Stretching": {"func": apply_histogram_stretching, "params": {}},
        "Histogram Equalization": {"func": apply_histogram_equalization, "params": {}},
    },
    "📢 Noise": {
        "Salt & Pepper Noise": {"func": apply_salt_and_pepper, "params": {
            "amount": {"type": "text", "placeholder": "0.01 to 0.50", "default": "0.04"},
            "salt_ratio": {"type": "text", "placeholder": "0.0 to 1.0", "default": "0.5"}
        }},
        "Gaussian Noise": {"func": apply_gaussian_noise, "params": {
            "var": {"type": "text", "placeholder": "0.001 to 0.10", "default": "0.01"}
        }},
        
        "Periodic Noise": {"func": apply_periodic_noise, "params": {
            "freq_h": {"type": "text", "placeholder": "0.01 to 0.50", "default": "0.05"},
            "freq_v": {"type": "text", "placeholder": "0.01 to 0.50", "default": "0.05"},
            "amplitude": {"type": "text", "placeholder": "1 to 255", "default": "30"}
        }},
        
        "Poisson Noise": {"func": apply_poisson_noise, "params": {}},
    },
    "🔄 Geometric": {
        "Rotate": {"func": apply_rotate, "params": {"angle": (0, 360, 1)}},
        
        "Resize": {"func": apply_resize, "params": {"scale_percent": "text"}},
        "Flip Horizontal": {"func": apply_flip_horizontal, "params": {}},
        "Flip Vertical": {"func": apply_flip_vertical, "params": {}},
    },
    "🔧 Filters": {
        "Threshold": {"func": apply_threshold, "params": {
            "thresh": {"type": "text", "placeholder": "0 to 255", "default": "127"}
        }},
        "Gaussian Blur": {"func": apply_blur, "params": {"kernel_size": (3, 31, 2)}},
        "Sharpen": {"func": apply_sharpen, "params": {}},
        
        
        
    },
        "✂️ Edge Detection": {
        "Canny": {"func": apply_canny, "params": {}},
        "Roberts": {"func": apply_roberts, "params": {}},
        "Prewitt": {"func": apply_prewitt, "params": {}},
        "Sobel": {"func": apply_sobel, "params": {}},
        "Laplacian": {"func": apply_laplacian, "params": {
            "alpha": {"type": "text", "placeholder": "0.0 to 1.0", "default": "0.2"}
        }},
        "Laplacian of Gaussian": {"func": apply_laplacian_of_gaussian, "params": {
            "sigma": {"type": "text", "placeholder": "0.5 to 10.0", "default": "2.0"}
        }},
    },
    "🔲 Morphological": {
        "Dilate": {"func": apply_dilate, "params": {
            "kernel_size": {"type": "text", "placeholder": "3 to 21 (odd)", "default": "5"},
            "iterations": {"type": "text", "placeholder": "1 to 20", "default": "1"}
        }},
        "Erode": {"func": apply_erode, "params": {
            "kernel_size": {"type": "text", "placeholder": "3 to 21 (odd)", "default": "5"},
            "iterations": {"type": "text", "placeholder": "1 to 20", "default": "1"}
        }},
        "Open": {"func": apply_open, "params": {
            "kernel_size": {"type": "text", "placeholder": "3 to 21 (odd)", "default": "5"}
        }},
        "Close": {"func": apply_close, "params": {
            "kernel_size": {"type": "text", "placeholder": "3 to 21 (odd)", "default": "5"}
        }},
    },
    "🎯 Clustering": {
        "K-Means": {"func": apply_kmeans, "params": {"k": (2, 10, 1), "recolor_mode": ["vivid", "center", "mixed"]}},
        "Fuzzy C-Means": {"func": apply_fcm, "params": {"k": (2, 10, 1), "m": (1.1, 5.0, 0.1), "recolor_mode": ["vivid", "center", "mixed"]}},
        "Probabilistic C-Means": {"func": apply_pcm, "params": {"k": (2, 10, 1), "recolor_mode": ["vivid", "center", "mixed"]}},
    },
}

# Flattens the nested structure into a single lookup dictionary.
EFFECTS = {}
for category in EFFECTS_CATEGORIES.values():
    EFFECTS.update(category)

# strips the numeric suffix to find the base effect name (from the dict id)
def parse_effect_id(effect_id):
    if effect_id in EFFECTS:
        return effect_id
    parts = effect_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base_name = parts[0]
        if base_name in EFFECTS:
            return base_name
    for key in EFFECTS.keys():
        if effect_id.startswith(key):
            return key
    raise KeyError(f"Could not find effect matching '{effect_id}'")

# iterate through the effects dict and apply the effects
def process_pipeline():
    if st.session_state.original_image is None:
        return None
    img = np.array(st.session_state.original_image)
    
    for effect_id, params in st.session_state.effects_pipeline.items():
        #skip disabled effects
        if effect_id in st.session_state.effects_disabled:
            continue
        
        effect_name = parse_effect_id(effect_id)
        effect_info = EFFECTS[effect_name]
        func = effect_info["func"]
        call_params = {}
        for param_name, param_value in params.items():
            call_params[param_name] = param_value
        img = func(img, **call_params)
    
    return Image.fromarray(img)

# check if img is grey or bw (true if 1 channel, or 3 identical channels, or only 0 and 255 values)
def is_grayscale_or_bw(img_array):
    
    if len(img_array.shape) == 2:
        return True
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Check if R == G == B (grayscale stored as RGB)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        if np.allclose(r, g) and np.allclose(g, b):
            return True
        
        # Check if binary (only 0 and 255)
        unique_vals = np.unique(img_array)
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 255}):
            return True
    
    return False


def create_histogram_figure(img_array, title, force_rgb=False):
    fig, ax = plt.subplots(figsize=(5, 2.8), dpi=100)
    
    # Determine if we should show RGB or BW histogram
    show_rgb = force_rgb or (len(img_array.shape) == 3 and not is_grayscale_or_bw(img_array))
    
    if show_rgb:
        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']
        for i, (color, label) in enumerate(zip(colors, labels)):
            hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=label, alpha=0.7, linewidth=1.5)
        ax.legend(loc='upper right', fontsize=7)
    else:
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color='white', linewidth=1.5)
    
    ax.set_xlim([0, 256])
    ax.set_xlabel('Pixel Value', fontsize=8, color='white')
    ax.set_ylabel('Frequency', fontsize=8, color='white')
    ax.set_title(title, fontsize=9, color='white', pad=5)
    ax.tick_params(colors='white', labelsize=7)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a1a')
    plt.tight_layout()
    return fig



# sidebar

with st.sidebar:
    tab_toolbox, tab_pipeline = st.tabs(["🧰 Toolbox", "📋 Active Effects"])

    with tab_toolbox:
        st.markdown(
            '<div style="font-size: 1.0em; font-weight: bold; color: #ff9f43; margin-bottom: 8px;">Browse & Add Effects</div>',
            unsafe_allow_html=True
        )
        # effects list
        for category_name, category_effects in EFFECTS_CATEGORIES.items():
            with st.expander(f"{category_name}", expanded=False):
                for effect_name, effect_info in category_effects.items():
                    has_params = len(effect_info["params"]) > 0
                    
                    if not has_params:
                        # No parameters  show as a direct left aligned button
                        if st.button(
                            f"➕ {effect_name}",
                            key=f"add_direct_{effect_name}_{st.session_state.effect_counter}",
                            use_container_width=True,
                            
                            type="secondary"
                        ):
                            instance_id = f"{effect_name}_{st.session_state.effect_counter}"
                            st.session_state.effect_counter += 1
                            st.session_state.effects_pipeline[instance_id] = {}
                            st.rerun()
                    else:
                        # Has parameters show in expander for configuration
                        with st.expander(f"➕ {effect_name}", expanded=False):
                            params = {}
                            for param_name, param_config in effect_info["params"].items():
                                display_name = PARAM_DISPLAY_NAMES.get(param_name, param_name.replace("_", " ").title())
                                
                                # if param is a range make a slider
                                if isinstance(param_config, tuple):
                                    min_val, max_val, step = param_config
                                    default = min_val + (max_val - min_val) // 2
                                    params[param_name] = st.slider(
                                        f"{display_name}",
                                        min_val, max_val, default, step,
                                        key=f"param_{effect_name}_{param_name}_{st.session_state.effect_counter}"
                                    )
                                # if param is a list make a selection box
                                elif isinstance(param_config, list):
                                    params[param_name] = st.selectbox(
                                        f"{display_name}",
                                        options=param_config,
                                        key=f"param_{effect_name}_{param_name}_{st.session_state.effect_counter}"
                                    )
                                # if param is a dict make an input box (multi param functions)
                                elif isinstance(param_config, dict) and param_config.get("type") == "text":
                                    placeholder = param_config.get("placeholder", "")
                                    default_val = param_config.get("default", "")
                                    params[param_name] = st.text_input(
                                        f"{display_name}",
                                        value=default_val,
                                        placeholder=placeholder,
                                        key=f"param_{effect_name}_{param_name}_{st.session_state.effect_counter}"
                                    )
                                # if param is a dict make an input box (single param functions)
                                elif param_config == "text":
                                    params[param_name] = st.text_input(
                                        f"{display_name}",
                                        value="100",
                                        key=f"param_{effect_name}_{param_name}_{st.session_state.effect_counter}"
                                    )
                                else:
                                    params[param_name] = param_config
                                    
                            # add the function to the active effects dict
                            if st.button(f"Add {effect_name}", key=f"add_{effect_name}_{st.session_state.effect_counter}", use_container_width=True):
                                instance_id = f"{effect_name}_{st.session_state.effect_counter}"
                                st.session_state.effect_counter += 1
                                st.session_state.effects_pipeline[instance_id] = params
                                st.rerun()
    # active effects tab
    with tab_pipeline:
        st.markdown(
            '<div style="font-size: 1.0em; font-weight: bold; color: #54a0ff; margin-bottom: 8px;">Manage Applied Effects</div>',
            unsafe_allow_html=True
        )

        if not st.session_state.effects_pipeline:
            st.info("No effects applied yet. Switch to the toolbox to add some.")
        else:
            active_count = len(st.session_state.effects_pipeline) - len(st.session_state.effects_disabled)
            st.success(f"{active_count} active / {len(st.session_state.effects_pipeline)} total")

            effects_to_remove = []
            effects_to_toggle = []
            
            for idx, (effect_id, params) in enumerate(st.session_state.effects_pipeline.items()):
                effect_name = parse_effect_id(effect_id)
                is_disabled = effect_id in st.session_state.effects_disabled
                
                opacity = "0.5" if is_disabled else "1.0"
                status_icon = "⏸️" if is_disabled else "▶️"
                
                col1, col2, col3 = st.columns([5, 1, 1])
                
                # display effect and params
                with col1:
                    param_str = ", ".join([
                        f"{PARAM_DISPLAY_NAMES.get(k, k)}={v:.3f}" if isinstance(v, float) else f"{PARAM_DISPLAY_NAMES.get(k, k)}={v}"
                        for k, v in params.items()
                    ])
                    if param_str:
                        st.markdown(
                            f'<div style="opacity: {opacity};"><b>{idx+1}. {status_icon} {effect_name}</b><br><code>{param_str}</code></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div style="opacity: {opacity};"><b>{idx+1}. {status_icon} {effect_name}</b></div>',
                            unsafe_allow_html=True
                        )
                        
                # display toggle status (on or off) + toggle on or off
                with col2:
                    toggle_label = "▶️" if is_disabled else "⏸️"
                    toggle_help = "Enable" if is_disabled else "Disable"
                    if st.button(toggle_label, key=f"toggle_{effect_id}", help=toggle_help):
                        effects_to_toggle.append(effect_id)
                # remove effect
                with col3:
                    if st.button("🗑️", key=f"remove_{effect_id}", help="Remove this effect"):
                        effects_to_remove.append(effect_id)
            
            # remove and toggle on/off the effects
            for effect_id in effects_to_toggle:
                if effect_id in st.session_state.effects_disabled:
                    st.session_state.effects_disabled.remove(effect_id)
                else:
                    st.session_state.effects_disabled.add(effect_id)
                st.rerun()

            for effect_id in effects_to_remove:
                del st.session_state.effects_pipeline[effect_id]
                st.session_state.effects_disabled.discard(effect_id)
                st.rerun()

            if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
                st.session_state.effects_pipeline.clear()
                st.session_state.effects_disabled.clear()
                st.rerun()

            st.markdown("---")
            st.caption("▶️ Active | ⏸️ Disabled (skipped in processing)")

# main page

st.title("🖼️ Image Processing Studio")
st.markdown("Upload an image and apply effects from the left sidebar.")

uploaded_file = st.file_uploader(
    "Drop your image here or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    accept_multiple_files=False,
    help="Supports PNG, JPG, JPEG, BMP, TIFF, WEBP"
)

# process uploaded file
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
#init the variable
    if st.session_state.original_image is None:
        st.session_state.original_image = image
        st.session_state.processed_image = image
# update existing variable + remove all effects
    elif uploaded_file.name != getattr(st.session_state, 'current_filename', None):
        st.session_state.original_image = image
        st.session_state.effects_pipeline.clear()
        st.session_state.effects_disabled.clear()
        st.session_state.current_filename = uploaded_file.name

# apply the effects
if st.session_state.original_image is not None:
    st.session_state.processed_image = process_pipeline()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📥 Original")
    if st.session_state.original_image is not None:
        st.image(st.session_state.original_image, use_container_width=True)
    else:
        st.markdown("""
        <div class="image-container">
            <div>
                <h3>📤 Drop Image Here</h3>
                <p>Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### ✨ Processed")
    if st.session_state.processed_image is not None:
        st.image(st.session_state.processed_image, use_container_width=True)

        buf = io.BytesIO()
        st.session_state.processed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Processed Image",
            data=byte_im,
            file_name="processed_image.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="image-container">
            <div>
                <h3>🎨 Apply Effects</h3>
                <p>Use the left sidebar to add image processing effects</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# stats
if st.session_state.original_image is not None and st.session_state.processed_image is not None:
    st.markdown("---")
    
    # comparison metrics
    mse = compute_mse(st.session_state.original_image, st.session_state.processed_image)
    psnr = compute_psnr(st.session_state.original_image, st.session_state.processed_image)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Mean Squared Error</div>
            <div class="metric-value">{:.2f}</div>
        </div>
        """.format(mse), unsafe_allow_html=True)
    with m_col2:
        psnr_display = "∞" if psnr == float('inf') else f"{psnr:.2f} dB"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Peak Signal-to-Noise Ratio</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(psnr_display), unsafe_allow_html=True)
    with m_col3:
        active_count = len(st.session_state.effects_pipeline) - len(st.session_state.effects_disabled)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Effects Active</div>
            <div class="metric-value">{}/{}</div>
        </div>
        """.format(active_count, len(st.session_state.effects_pipeline)), unsafe_allow_html=True)

    #histograms
    st.markdown("### 📈 Histograms")
    
    orig_arr = np.array(st.session_state.original_image)
    proc_arr = np.array(st.session_state.processed_image)
    
    # Determine if processed image is grayscale/BW
    proc_is_bw = is_grayscale_or_bw(proc_arr)
    orig_is_bw = is_grayscale_or_bw(orig_arr)

    h_col1, h_col2 = st.columns(2)
    with h_col1:
        fig_orig = create_histogram_figure(orig_arr, "Original Image", force_rgb=not orig_is_bw)
        st.pyplot(fig_orig, use_container_width=True)
    with h_col2:
        fig_proc = create_histogram_figure(proc_arr, "Processed Image", force_rgb=not proc_is_bw)
        st.pyplot(fig_proc, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit & OpenCV | Image Processing Class")