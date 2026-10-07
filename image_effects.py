import numpy as np
import cv2
import streamlit as st
from scipy import ndimage


# color effects

def apply_grayscale(img_array):
    if len(img_array.shape) == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return img_array



def apply_invert(img_array):
    return cv2.bitwise_not(img_array)


def apply_brightness(img_array, value=30):
    return cv2.convertScaleAbs(img_array, alpha=1, beta=value)


# contrast scaling (multiply)
def apply_contrast(img_array, alpha=1.5):
    return cv2.convertScaleAbs(img_array, alpha=alpha, beta=0)

#  histogram stretching, map 2nd percentile, 98th percentile to 0, 255
def apply_histogram_stretching(img_array):
    
    result = img_array.copy().astype(np.float32)
    
    if len(img_array.shape) == 3:
        # per-channel stretching for RGB
        for i in range(3):
            channel = result[:, :, i]
            low_val = np.percentile(channel, 2)
            high_val = np.percentile(channel, 98)
            
            # avoid division by zero
            if high_val - low_val < 1e-6:
                continue
                
            # Stretch to full [0, 255] range
            stretched = (channel - low_val) * (255.0 / (high_val - low_val))
            result[:, :, i] = np.clip(stretched, 0, 255)
    else:
        # grayscale
        low_val = np.percentile(result, 2)
        high_val = np.percentile(result, 98)
        
        if high_val - low_val > 1e-6:
            stretched = (result - low_val) * (255.0 / (high_val - low_val))
            result = np.clip(stretched, 0, 255)
    
    return result.astype(np.uint8)


def apply_histogram_equalization(img_array):
    # special equalization for RGB Y is brightness, red chroma (red vs green) and blue chroma (blue vs yellow)
    if len(img_array.shape) == 3:
        ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        # only equlize brightness
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    # equalize the entire image if greyscale
    return cv2.equalizeHist(img_array)


# noise effects

def apply_salt_and_pepper(img_array, amount=0.04, salt_ratio=0.5):
     
    amount = float(amount)
    salt_ratio = float(salt_ratio)

    noisy = img_array.copy()
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    num_salt = int(amount * total_pixels * salt_ratio)
    num_pepper = int(amount * total_pixels * (1 - salt_ratio))
    
    # salt noise
    coords = [np.random.randint(0, i, num_salt) for i in img_array.shape[:2]]
    # if rgb make all channels 255 else greyscale make the only channel value 255
    if len(img_array.shape) == 3:
        noisy[coords[0], coords[1], :] = 255
    else:
        noisy[coords[0], coords[1]] = 255
    
    # pepper noise
    coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape[:2]]
    if len(img_array.shape) == 3:
        noisy[coords[0], coords[1], :] = 0
    else:
        noisy[coords[0], coords[1]] = 0
    
    return noisy


def apply_gaussian_noise(img_array, mean=0, var=0.01):
    var = float(var)
    sigma = var ** 0.5
    # get a random normal value using passed parameters
    gauss = np.random.normal(mean, sigma, img_array.shape)
    # apply to image
    noisy = img_array.astype(np.float64) + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_poisson_noise(img_array):
    # scale to 0, 1 then apply Poisson
    vals = len(np.unique(img_array))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img_array.astype(np.float64) * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_periodic_noise(img_array, freq_h=0.05, freq_v=0.05, amplitude=30):

    freq_h = float(freq_h)
    freq_v = float(freq_v)
    amplitude = float(amplitude)
    rows, cols = img_array.shape[:2]
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # generate sinusoidal pattern
    noise_pattern = amplitude * (np.sin(2 * np.pi * freq_h * X) + np.sin(2 * np.pi * freq_v * Y))
    
    if len(img_array.shape) == 3:
        noise_pattern = np.stack([noise_pattern] * 3, axis=-1)
    
    noisy = img_array.astype(np.float64) + noise_pattern
    return np.clip(noisy, 0, 255).astype(np.uint8)



#geometric effects

def apply_rotate(img_array, angle=90):
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_array, M, (w, h))


def apply_flip_horizontal(img_array):
    return cv2.flip(img_array, 1)


def apply_flip_vertical(img_array):
    return cv2.flip(img_array, 0)


# filters

def apply_blur(img_array, kernel_size=5):
    return cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)


def apply_sharpen(img_array):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if len(img_array.shape) == 2:
        return cv2.filter2D(img_array, -1, kernel)
    return cv2.filter2D(img_array, -1, kernel)



def apply_threshold(img_array, thresh=127):

    thresh = int(float(thresh))
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) if len(img_array.shape) == 3 else binary


def apply_otsu_threshold(img_array):

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Otsu's method automatically finds optimal threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) if len(img_array.shape) == 3 else binary


# edges


def _get_gray(img_array):
    if len(img_array.shape) == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return img_array.astype(np.float32)

# for shape consistency
def _edges_to_rgb(edges, original_shape):
    if len(original_shape) == 3:
        return cv2.cvtColor(edges.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return edges.astype(np.uint8)


def apply_canny(img_array):
    gray = _get_gray(img_array)
    median = np.median(gray)
    low = int(max(0, 0.67 * median))
    high = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray.astype(np.uint8), low, high)
    return _edges_to_rgb(edges, img_array.shape)


def apply_roberts(img_array):
    gray = _get_gray(img_array)
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, roberts_x)
    gy = cv2.filter2D(gray, -1, roberts_y)
    edges = np.sqrt(gx**2 + gy**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return _edges_to_rgb(edges, img_array.shape)


def apply_prewitt(img_array):
    gray = _get_gray(img_array)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, -1, prewitt_x)
    gy = cv2.filter2D(gray, -1, prewitt_y)
    edges = np.sqrt(gx**2 + gy**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return _edges_to_rgb(edges, img_array.shape)


def apply_sobel(img_array):
    gray = _get_gray(img_array)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return _edges_to_rgb(edges, img_array.shape)


def apply_laplacian(img_array, alpha=0.2):
    """
    Laplacian edge detection using MATLAB-style fspecial('laplacian', alpha) kernel.
    alpha: 0.0 to 1.0. Controls shape of Laplacian (0 = standard 4-neighbor, 1 = 8-neighbor)
    """
    alpha = float(alpha)
    gray = _get_gray(img_array)
    
    # Build the kernel from the formula
    kernel = np.array([
        [alpha/4, (1-alpha)/4, alpha/4],
        [(1-alpha)/4, -1, (1-alpha)/4],
        [alpha/4, (1-alpha)/4, alpha/4]
    ], dtype=np.float32)
    kernel = kernel * (4.0 / (alpha + 1))
    
    edges = cv2.filter2D(gray, -1, kernel)
    edges = np.abs(edges)  
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    return _edges_to_rgb(edges, img_array.shape)


def apply_laplacian_of_gaussian(img_array, sigma=2.0):
   
    sigma = float(sigma)
    gray = _get_gray(img_array)
  
    # Normalize to 0,1 for processing
    gray_norm = gray / 255.0
    log_edges = ndimage.gaussian_laplace(gray_norm, sigma=sigma)
    
    # Take absolute value and scale back to 0,25
    edges = np.abs(log_edges)
    edges = (edges / (edges.max() + 1e-10)) * 255  
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    
    return _edges_to_rgb(edges, img_array.shape)


# morph

def apply_dilate(img_array, kernel_size=5, iterations=1):
    kernel_size = int(float(kernel_size))
    iterations = int(float(iterations))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if len(img_array.shape) == 3:
        return cv2.dilate(img_array, kernel, iterations=iterations)
    return cv2.dilate(img_array, kernel, iterations=iterations)


def apply_erode(img_array, kernel_size=5, iterations=1):
    kernel_size = int(float(kernel_size))
    iterations = int(float(iterations))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if len(img_array.shape) == 3:
        return cv2.erode(img_array, kernel, iterations=iterations)
    return cv2.erode(img_array, kernel, iterations=iterations)


def apply_open(img_array, kernel_size=5):
    kernel_size = int(float(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if len(img_array.shape) == 3:
        return cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)


def apply_close(img_array, kernel_size=5):
    kernel_size = int(float(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if len(img_array.shape) == 3:
        return cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)


# clustering

CLUSTER_COLORS = np.array([
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [255, 0, 255],    # Magenta
    [0, 255, 255],    # Cyan
    [255, 128, 0],    # Orange
    [128, 0, 255],    # Purple
    [255, 192, 203],  # Pink
    [0, 128, 0],      # Dark Green
], dtype=np.uint8)


# coloring options, center color, preset vivid colors, mix of both
def _recolor_with_centers(labels, centers, original_shape):
    centers = np.uint8(centers)
    segmented = centers[labels]
    return segmented.reshape(original_shape)


def _recolor_with_vivid_colors(labels, k, original_shape):
    colors = CLUSTER_COLORS[:k]
    segmented = colors[labels]
    return segmented.reshape(original_shape)


def _recolor_mixed(labels, centers, k, original_shape, mix_ratio=0.6):
    centers = np.uint8(centers)
    vivid = CLUSTER_COLORS[:k]
    blended = (mix_ratio * centers + (1 - mix_ratio) * vivid).astype(np.uint8)
    segmented = blended[labels]
    return segmented.reshape(original_shape)


#clustering functions

def apply_kmeans(img_array, k=3, recolor_mode="vivid"):
    # vectorize
    pixels = img_array.reshape((-1, 3)).astype(np.float32)
    
    #show spinner to display calculation progress (if the image is big the site might look frozen to the user while they wait for processing)
    with st.spinner(f"Running K-Means (k={k}) on {img_array.shape[0]}×{img_array.shape[1]} image..."):
        #stop at max iter 100 or absolute movement of center below 0.2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        
        if recolor_mode == "center":
            result = _recolor_with_centers(labels, centers, img_array.shape)
        elif recolor_mode == "mixed":
            result = _recolor_mixed(labels, centers, k, img_array.shape)
        else:
            result = _recolor_with_vivid_colors(labels, k, img_array.shape)
    
    return result


## found equivalents of these functions online and tweaked them
def apply_fcm(img_array, k=3, m=2.0, max_iter=100, epsilon=0.01, recolor_mode="vivid"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pixels = img_array.reshape((-1, 3)).astype(np.float32)
    n_samples = pixels.shape[0]
    
    np.random.seed(42)
    indices = np.random.choice(n_samples, k, replace=False)
    centers = pixels[indices].copy()
    
    for iteration in range(max_iter):
        status_text.text(f"FCM iteration {iteration + 1}/{max_iter} | {n_samples:,} pixels | {k} clusters")
        progress_bar.progress(min((iteration + 1) / max_iter, 0.99))
        
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(pixels - centers[i], axis=1)
        
        distances = np.maximum(distances, 1e-10)
        
        power = 2.0 / (m - 1)
        membership = np.zeros((n_samples, k))
        for i in range(k):
            denom = np.sum((distances[:, i:i+1] / distances) ** power, axis=1)
            membership[:, i] = 1.0 / denom
        
        new_centers = np.zeros_like(centers)
        for i in range(k):
            um = membership[:, i:i+1] ** m
            new_centers[i] = np.sum(um * pixels, axis=0) / np.sum(um)
        
        if np.linalg.norm(new_centers - centers) < epsilon:
            status_text.text(f"FCM converged at iteration {iteration + 1}")
            break
        centers = new_centers
    
    progress_bar.empty()
    status_text.empty()
    
    labels = np.argmax(membership, axis=1)
    
    if recolor_mode == "center":
        result = _recolor_with_centers(labels, centers, img_array.shape)
    elif recolor_mode == "mixed":
        result = _recolor_mixed(labels, centers, k, img_array.shape)
    else:
        result = _recolor_with_vivid_colors(labels, k, img_array.shape)
    
    return result


def apply_pcm(img_array, k=3, eta=None, max_iter=100, epsilon=0.01, recolor_mode="vivid"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    pixels = img_array.reshape((-1, 3)).astype(np.float32)
    n_samples = pixels.shape[0]
    
    status_text.text("PCM: Initializing with K-Means...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.float32)
    
    if eta is None:
        eta = np.zeros(k)
        for i in range(k):
            cluster_pixels = pixels[labels.flatten() == i]
            if len(cluster_pixels) > 0:
                eta[i] = np.mean(np.linalg.norm(cluster_pixels - centers[i], axis=1) ** 2)
            else:
                eta[i] = 1000.0
    
    for iteration in range(max_iter):
        status_text.text(f"PCM iteration {iteration + 1}/{max_iter} | {n_samples:,} pixels | {k} clusters")
        progress_bar.progress(min((iteration + 1) / max_iter, 0.99))
        
        distances = np.zeros((n_samples, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(pixels - centers[i], axis=1)
        
        distances = np.maximum(distances, 1e-10)
        
        membership = np.zeros((n_samples, k))
        for i in range(k):
            membership[:, i] = np.exp(-distances[:, i]**2 / eta[i])
        
        new_centers = np.zeros_like(centers)
        for i in range(k):
            denom = np.sum(membership[:, i])
            if denom > 0:
                new_centers[i] = np.sum(membership[:, i:i+1] * pixels, axis=0) / denom
            else:
                new_centers[i] = centers[i]
        
        for i in range(k):
            cluster_distances = distances[:, i] ** 2
            weighted_dist = membership[:, i] * cluster_distances
            denom = np.sum(membership[:, i])
            if denom > 0:
                eta[i] = np.sum(weighted_dist) / denom
        
        if np.linalg.norm(new_centers - centers) < epsilon:
            status_text.text(f"PCM converged at iteration {iteration + 1}")
            break
        centers = new_centers
    
    progress_bar.empty()
    status_text.empty()
    
    labels = np.argmax(membership, axis=1)
    
    if recolor_mode == "center":
        result = _recolor_with_centers(labels, centers, img_array.shape)
    elif recolor_mode == "mixed":
        result = _recolor_mixed(labels, centers, k, img_array.shape)
    else:
        result = _recolor_with_vivid_colors(labels, k, img_array.shape)
    
    return result


# metrics

def compute_mse(img1, img2):
    arr1 = np.array(img1).astype(np.float64)
    arr2 = np.array(img2).astype(np.float64)
    
    if len(arr1.shape) != len(arr2.shape):
        if len(arr1.shape) == 3:
            arr1 = cv2.cvtColor(arr1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
        if len(arr2.shape) == 3:
            arr2 = cv2.cvtColor(arr2.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    if arr1.shape != arr2.shape:
        h, w = arr2.shape[:2]
        if len(arr1.shape) == 3:
            arr1 = cv2.resize(arr1.astype(np.uint8), (w, h))
            arr1 = arr1.astype(np.float64)
        else:
            arr1 = cv2.resize(arr1.astype(np.uint8), (w, h))
            arr1 = arr1.astype(np.float64)
    
    mse = np.mean((arr1 - arr2) ** 2)
    return mse


def compute_psnr(img1, img2):
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def apply_resize(img_array, scale_percent=100):
    
    scale_percent = float(scale_percent)
    scale = max(10, min(400, scale_percent)) / 100.0
    width = int(img_array.shape[1] * scale)
    height = int(img_array.shape[0] * scale)
    return cv2.resize(img_array, (width, height), interpolation=cv2.INTER_LINEAR)