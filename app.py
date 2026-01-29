from flask import Flask, request, send_file, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import os
import cv2
import base64
import requests
import zipfile
import tempfile
import ftplib
import uuid
from datetime import datetime

app = Flask(__name__)

# Environment variables
API_KEY = os.environ.get('API_KEY', None)
FAL_KEY = os.environ.get('FAL_KEY', None)  # For image enhancement

# FTP Configuration
FTP_HOST = os.environ.get('FTP_HOST', None)
FTP_USER = os.environ.get('FTP_USER', None)
FTP_PASS = os.environ.get('FTP_PASS', None)
FTP_DIR = os.environ.get('FTP_DIR', '/logos')  # Remote directory to upload to
FTP_BASE_URL = os.environ.get('FTP_BASE_URL', None)  # Public URL base (e.g., https://cdn.example.com/logos)

MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE_MB', 10)) * 1024 * 1024
DEFAULT_THRESHOLD = int(os.environ.get('DEFAULT_THRESHOLD', 100))
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def verify_api_key():
    """Verify API key if set in environment"""
    if API_KEY:
        provided_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        if provided_key != API_KEY:
            return jsonify({
                "error": "Unauthorized",
                "message": "Invalid or missing API key"
            }), 401
    return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "Image Color Replacer API",
        "status": "running",
        "version": "3.1.0",
        "authentication": "required" if API_KEY else "not required",
        "endpoints": {
            "/health": "GET - Health check",
            "/remove-bg": "POST - 🎯 Automatic background removal (remove.bg quality)",
            "/remove-bg/info": "GET - Info about background removal endpoint",
            "/process-logo": "POST - 🚀 FULL PIPELINE: enhance → remove bg → trim → generate all variants",
            "/gen-logo-variant": "POST - 🎯 Generate color variant (with enhance, bg removal, trim)",
            "/logo-type": "POST - 🔍 Detect logo type (returns 'black' or 'white')",
            "/smart-print-ready": "POST - SMART print-ready conversion",
            "/smart-print-ready/analyze": "POST - Analyze logo before processing",
            "/smart-logo-variant": "POST - Generate logo variant (outline fallback)",
            "/force-solid-black": "POST - Force entire logo to solid black",
            "/force-solid-white": "POST - Force entire logo to solid white",
            "/replace-dark-to-white": "POST - Replace dark colors with white",
            "/replace-light-to-dark": "POST - Replace light colors with dark",
            "/invert-colors": "POST - Invert all colors",
            "/check-transparency": "POST - Check image transparency"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "message": "Service is running"})


# ============================================================
# REMOVE-BG: Professional Background Removal (remove.bg quality)
# ============================================================

def analyze_image_for_bg_removal(img):
    """
    Analyze image characteristics to determine best removal strategy
    Returns dict with analysis results
    """
    img_rgb = img.convert('RGB')
    data = np.array(img_rgb)
    h, w = data.shape[:2]
    
    analysis = {
        'has_solid_bg': False,
        'bg_color': None,
        'bg_coverage': 0,
        'is_graphic': False,
        'color_complexity': 0,
        'edge_sharpness': 0
    }
    
    # 1. Analyze corners for solid background
    corner_size = max(5, min(30, h // 15, w // 15))
    
    corners = [
        data[0:corner_size, 0:corner_size],
        data[0:corner_size, w-corner_size:w],
        data[h-corner_size:h, 0:corner_size],
        data[h-corner_size:h, w-corner_size:w],
    ]
    
    # Also sample edge midpoints
    edge_samples = [
        data[0:corner_size, w//2-corner_size//2:w//2+corner_size//2],  # top
        data[h-corner_size:h, w//2-corner_size//2:w//2+corner_size//2],  # bottom
        data[h//2-corner_size//2:h//2+corner_size//2, 0:corner_size],  # left
        data[h//2-corner_size//2:h//2+corner_size//2, w-corner_size:w],  # right
    ]
    
    all_border_samples = corners + edge_samples
    
    # Calculate color stats for each sample
    sample_means = []
    sample_stds = []
    for sample in all_border_samples:
        pixels = sample.reshape(-1, 3)
        sample_means.append(np.mean(pixels, axis=0))
        sample_stds.append(np.std(pixels))
    
    # Check if borders have consistent color (solid background indicator)
    mean_of_means = np.mean(sample_means, axis=0)
    std_between_samples = np.std(sample_means, axis=0).mean()
    avg_internal_std = np.mean(sample_stds)
    
    # Solid background: low variation within samples AND between samples
    if avg_internal_std < 20 and std_between_samples < 25:
        analysis['has_solid_bg'] = True
        analysis['bg_color'] = mean_of_means.astype(np.uint8)
        
        # Calculate how much of image is this background color
        tolerance = 30
        bg_mask = np.all(np.abs(data.astype(np.int16) - analysis['bg_color'].astype(np.int16)) < tolerance, axis=2)
        analysis['bg_coverage'] = np.sum(bg_mask) / (h * w)
    
    # 2. Analyze color complexity (logos typically have fewer colors)
    # Quantize to reduce noise
    quantized = (data // 32) * 32
    unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
    max_possible = (h * w)
    analysis['color_complexity'] = unique_colors / max_possible
    
    # Graphics/logos typically have < 5% color complexity
    if analysis['color_complexity'] < 0.05:
        analysis['is_graphic'] = True
    
    # 3. Analyze edge sharpness (graphics have sharp edges, photos have gradients)
    gray = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    analysis['edge_sharpness'] = np.var(laplacian)
    
    return analysis


def remove_bg_color_method(img, bg_color=None, tolerance=25):
    """
    Color-based background removal - perfect for logos/graphics
    
    KEY PRINCIPLE: Only remove pixels that are:
    1. Similar to background color AND
    2. Connected to the image border (via flood fill)
    
    This ensures we NEVER remove interior content, even if it's 
    similar in color to the background (like white text on gray bg)
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    h, w = data.shape[:2]
    rgb = data[:, :, :3].astype(np.float32)
    
    # Detect background color if not provided
    if bg_color is None:
        corner_size = max(5, min(20, h // 15, w // 15))
        corners = [
            rgb[0:corner_size, 0:corner_size],
            rgb[0:corner_size, w-corner_size:w],
            rgb[h-corner_size:h, 0:corner_size],
            rgb[h-corner_size:h, w-corner_size:w],
        ]
        all_corner_pixels = np.vstack([c.reshape(-1, 3) for c in corners])
        bg_color = np.median(all_corner_pixels, axis=0)
    
    bg_color = np.array(bg_color, dtype=np.float32)
    
    # Calculate color distance from background for each pixel
    color_diff = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))
    
    # Create binary mask: pixels that COULD be background (within tolerance)
    potential_bg = (color_diff < tolerance).astype(np.uint8) * 255
    
    # Use flood fill from corners and edge midpoints
    work = potential_bg.copy()
    
    # Flood fill from corners
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        if work[seed[1], seed[0]] > 0:
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(work, mask, seed, 128)
    
    # Flood fill from edge midpoints
    for seed in [(w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)]:
        if work[seed[1], seed[0]] > 0:
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(work, mask, seed, 128)
    
    # Additional flood fills along edges for complete coverage
    step = max(1, min(w, h) // 20)
    for x in range(0, w, step):
        for y_seed in [0, h-1]:
            if work[y_seed, x] > 0:
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(work, mask, (x, y_seed), 128)
    for y in range(0, h, step):
        for x_seed in [0, w-1]:
            if work[y, x_seed] > 0:
                mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                cv2.floodFill(work, mask, (x_seed, y), 128)
    
    # Connected background is where we flood filled to value 128
    connected_bg = (work == 128).astype(np.uint8) * 255
    
    # Create alpha channel - start fully opaque
    alpha = np.ones((h, w), dtype=np.float32) * 255
    
    # Only the connected background becomes transparent
    alpha[connected_bg > 0] = 0
    
    # Anti-alias the edges for smooth transitions
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(connected_bg, kernel, iterations=1)
    edge_mask = (dilated > 0) & (connected_bg == 0)
    
    # For edge pixels, create soft alpha based on color distance
    edge_alpha = np.clip(color_diff / tolerance * 255, 0, 255)
    alpha[edge_mask] = np.minimum(alpha[edge_mask], edge_alpha[edge_mask])
    
    # Slight Gaussian blur for smoother edges
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
    
    # Apply alpha to image
    result = data.copy()
    result[:, :, 3] = alpha.astype(np.uint8)
    
    return Image.fromarray(result, 'RGBA')


def remove_bg_ai_method(img_bytes, model='isnet-general-use'):
    """
    AI-based background removal using rembg
    Best for photos with complex backgrounds
    """
    try:
        from rembg import remove, new_session
        
        session = new_session(model)
        output_bytes = remove(
            img_bytes,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            post_process_mask=True
        )
        
        return Image.open(BytesIO(output_bytes)).convert('RGBA'), True
        
    except ImportError:
        return None, False
    except Exception as e:
        return None, False


def count_content_pixels(img_rgba, min_alpha=20):
    """Count pixels that have meaningful content (not transparent)"""
    data = np.array(img_rgba)
    return np.sum(data[:, :, 3] > min_alpha)


def remove_bg_smart(img_bytes):
    """
    Smart background removal that automatically chooses the best method
    Similar to how remove.bg works - tries to preserve all content
    
    Strategy:
    1. Analyze image characteristics
    2. For solid backgrounds: Use color-based (preserves all content)
    3. For complex backgrounds: Use AI model
    4. Validate result has reasonable content preserved
    """
    img = Image.open(BytesIO(img_bytes))
    original_pixels = img.width * img.height
    
    # Analyze image
    analysis = analyze_image_for_bg_removal(img)
    
    method_used = None
    result_img = None
    
    # Decision logic - prefer color-based for solid backgrounds
    use_color_method = (
        analysis['has_solid_bg'] and 
        analysis['bg_coverage'] > 0.10  # At least 10% is background
    ) or (
        analysis['is_graphic'] and 
        analysis['color_complexity'] < 0.05
    )
    
    if use_color_method:
        # Determine tolerance based on background color
        # For light backgrounds (like gray/white), use LOWER tolerance
        # to avoid removing light-colored content
        bg_color = analysis.get('bg_color')
        if bg_color is not None:
            bg_brightness = np.mean(bg_color)
            if bg_brightness > 200:  # Very light background
                base_tolerance = 15
            elif bg_brightness > 150:  # Light background  
                base_tolerance = 18
            elif bg_brightness < 50:  # Very dark background
                base_tolerance = 15
            else:  # Medium brightness
                base_tolerance = 22
        else:
            base_tolerance = 20
        
        # Try color-based removal with calculated tolerance
        result_img = remove_bg_color_method(
            img, 
            bg_color=analysis['bg_color'],
            tolerance=base_tolerance
        )
        method_used = f"color-based (tol={base_tolerance})"
        
        # Validate: check if we preserved reasonable content
        content_pixels = count_content_pixels(result_img)
        content_ratio = content_pixels / original_pixels
        
        # If too little content (< 3%) or too much (> 98%), adjust tolerance
        if content_ratio < 0.03:
            # Too aggressive - try lower tolerance
            for tol in [12, 10, 8]:
                alt_result = remove_bg_color_method(img, bg_color=analysis['bg_color'], tolerance=tol)
                alt_content = count_content_pixels(alt_result)
                alt_ratio = alt_content / original_pixels
                if alt_ratio > 0.03:
                    result_img = alt_result
                    method_used = f"color-based (tol={tol}, adjusted)"
                    break
        elif content_ratio > 0.98:
            # Not aggressive enough - try higher tolerance
            for tol in [25, 30, 35]:
                alt_result = remove_bg_color_method(img, bg_color=analysis['bg_color'], tolerance=tol)
                alt_content = count_content_pixels(alt_result)
                alt_ratio = alt_content / original_pixels
                if alt_ratio < 0.98:
                    result_img = alt_result
                    method_used = f"color-based (tol={tol}, adjusted)"
                    break
    
    else:
        # Try AI method for photos
        ai_result, ai_success = remove_bg_ai_method(img_bytes)
        
        if ai_success and ai_result:
            result_img = ai_result
            method_used = "AI (isnet-general-use)"
            
            # Validate AI result
            content_pixels = count_content_pixels(result_img)
            content_ratio = content_pixels / original_pixels
            
            # If AI removed too much (common with logos), fall back to color method
            if content_ratio < 0.10 and analysis['has_solid_bg']:
                color_result = remove_bg_color_method(img, bg_color=analysis['bg_color'], tolerance=20)
                color_content = count_content_pixels(color_result)
                
                # Use color method if it preserved more content
                if color_content > content_pixels * 1.3:
                    result_img = color_result
                    method_used = "color-based (AI fallback)"
        else:
            # AI not available, use color method
            result_img = remove_bg_color_method(img, tolerance=20)
            method_used = "color-based (AI unavailable)"
    
    # Final fallback
    if result_img is None:
        result_img = remove_bg_color_method(img, tolerance=20)
        method_used = "color-based (fallback)"
    
    return result_img, method_used, analysis


def has_transparency(img_bytes):
    """
    Check if image already has transparency (alpha channel with non-255 values)
    Returns: True if image has transparency, False otherwise
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        
        # If image doesn't have alpha channel, it's not transparent
        if img.mode not in ('RGBA', 'LA', 'PA'):
            return False
        
        # Convert to RGBA to standardize
        img_rgba = img.convert('RGBA')
        data = np.array(img_rgba)
        alpha = data[:, :, 3]
        
        # Check if any pixel has alpha < 255 (partially or fully transparent)
        has_transparent_pixels = np.any(alpha < 255)
        
        return has_transparent_pixels
    except Exception:
        return False


def refine_edges(img_rgba, edge_smoothing=1, feather_amount=0):
    """
    Post-process to refine edges and reduce jaggedness
    Similar to remove.bg's edge refinement
    """
    data = np.array(img_rgba)
    alpha = data[:, :, 3]
    
    if edge_smoothing > 0:
        # Smooth alpha channel edges
        alpha_float = alpha.astype(np.float32)
        
        # Apply slight Gaussian blur to alpha for smoother edges
        kernel_size = edge_smoothing * 2 + 1
        alpha_smooth = cv2.GaussianBlur(alpha_float, (kernel_size, kernel_size), 0)
        
        # Preserve fully opaque and fully transparent areas
        # Only smooth the transition zones
        mask = (alpha > 5) & (alpha < 250)
        alpha = alpha.astype(np.float32)
        alpha[mask] = alpha_smooth[mask]
        
        data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    
    if feather_amount > 0:
        # Feather edges for softer blending
        alpha = data[:, :, 3].astype(np.float32)
        
        # Create edge mask
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha, kernel, iterations=feather_amount)
        eroded = cv2.erode(alpha, kernel, iterations=feather_amount)
        edge_mask = (dilated - eroded) > 0
        
        # Apply gradient to edges
        blurred = cv2.GaussianBlur(alpha, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
        alpha[edge_mask] = blurred[edge_mask]
        
        data[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    
    return Image.fromarray(data, 'RGBA')


@app.route('/remove-bg', methods=['POST'])
def remove_bg_endpoint():
    """
    /remove-bg with full step logging (success + failure)

    Logs are returned in:
      - Header: X-Step-Log (short human readable)
      - Header: X-Step-Log-Json (base64 json, compact)
      - Error JSON: includes "processing_log"
    """
    import time
    import json
    import base64

    start_time = time.time()

    # ---- logging helpers ----
    processing_log = []  # list of dicts

    def t_ms():
        return int((time.time() - start_time) * 1000)

    def log(step, success=True, **data):
        entry = {"step": step, "success": bool(success), "t_ms": t_ms()}
        entry.update(data)
        processing_log.append(entry)

    def attach_logs_to_response(resp):
        # short header-friendly summary
        summary = []
        for e in processing_log:
            summary.append(f"{e['step']}:{'ok' if e.get('success') else 'fail'}@{e.get('t_ms')}ms")
        resp.headers["X-Step-Log"] = " | ".join(summary)

        # compact base64 json for machines
        b = json.dumps(processing_log, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        resp.headers["X-Step-Log-Json"] = base64.b64encode(b).decode("ascii")
        return resp

    def json_error(payload, status=400):
        # always include logs in error json
        payload["processing_log"] = processing_log
        resp = jsonify(payload)
        resp.status_code = status
        return attach_logs_to_response(resp)

    # ---- auth (log + always attach) ----
    log("auth_check", success=True)
    auth_error = verify_api_key()
    if auth_error:
        # verify_api_key returns (jsonify(...), 401)
        resp, status = auth_error
        resp.status_code = status
        log("auth_check", success=False, reason="invalid_or_missing_api_key")
        return attach_logs_to_response(resp)

    try:
        # ---- validate input ----
        if 'image' not in request.files:
            log("validate_input", success=False, reason="missing_image_file")
            return json_error({
                "error": "No image file provided",
                "usage": {
                    "endpoint": "/remove-bg",
                    "method": "POST",
                    "content_type": "multipart/form-data",
                    "required_field": "image",
                    "optional_fields": {
                        "enhance": "true | false (default=false)",
                        "trim": "true | false (default=true)",
                        "output_format": "png | webp",
                        "bg_remove": "auto | ai | color | skip (default=auto)"
                    }
                }
            }, status=400)

        file = request.files['image']
        log("validate_input", success=True, filename=getattr(file, "filename", ""))

        # ---- read params ----
        do_enhance = request.form.get('enhance', 'false').lower() == 'true'
        do_trim = request.form.get('trim', 'true').lower() == 'true'
        output_format = request.form.get('output_format', 'png').lower()

        bg_remove = request.form.get('bg_remove', 'auto').strip().lower()
        allowed_bg_remove = {'auto', 'ai', 'color', 'skip'}
        if bg_remove not in allowed_bg_remove:
            log("read_params", success=False, reason="invalid_bg_remove", received=bg_remove)
            return json_error({
                "error": "Invalid bg_remove value",
                "allowed": sorted(list(allowed_bg_remove)),
                "received": bg_remove
            }, status=400)

        log("read_params", success=True, enhance=do_enhance, trim=do_trim,
            output_format=output_format, bg_remove=bg_remove)

        # ---- read image bytes ----
        img_bytes = file.read()
        log("read_image", success=True, bytes=len(img_bytes))

        # ---- transparency check ----
        try:
            already_transparent = has_transparency(img_bytes)
            log("check_transparency", success=True, already_transparent=already_transparent)
        except Exception as e:
            # don’t kill the request; assume false and continue
            already_transparent = False
            log("check_transparency", success=False, error=str(e), already_transparent=False)

        # ---- STEP 1: enhance ----
        enhanced = False
        enhance_msg = "not requested"

        if do_enhance:
            try:
                enhanced_bytes, enhanced, enhance_msg = enhance_image_fal(img_bytes)
                if enhanced and enhanced_bytes:
                    img_bytes = enhanced_bytes
                    log("enhance_fal", success=True, applied=True, message=enhance_msg, new_bytes=len(img_bytes))
                else:
                    log("enhance_fal", success=True, applied=False, message=enhance_msg)
            except Exception as e:
                enhanced = False
                enhance_msg = f"failed: {e}"
                log("enhance_fal", success=False, applied=False, error=str(e))
        else:
            log("enhance_fal", success=True, applied=False, message="not requested")

        # ---- STEP 2: background removal ----
        analysis = {}
        fallback_used = False

        if bg_remove == 'skip':
            result_img = Image.open(BytesIO(img_bytes))
            if result_img.mode != 'RGBA':
                result_img = result_img.convert('RGBA')
            method_used = 'skipped (bg_remove=skip)'
            log("bg_remove", success=True, mode="skip", method_used=method_used)

        elif already_transparent:
            result_img = Image.open(BytesIO(img_bytes))
            if result_img.mode != 'RGBA':
                result_img = result_img.convert('RGBA')
            method_used = 'skipped (already transparent)'
            log("bg_remove", success=True, mode="already_transparent", method_used=method_used)

        else:
            if bg_remove == 'auto':
                result_img, method_used, analysis = remove_bg_smart(img_bytes)
                log("bg_remove_auto", success=True, method_used=method_used, **(analysis or {}))

            elif bg_remove == 'ai':
                ai_img, ai_ok = remove_bg_ai_method(img_bytes)
                if ai_ok and ai_img is not None:
                    result_img = ai_img
                    method_used = 'AI (forced)'
                    analysis = {"forced": "ai"}
                    log("bg_remove_ai", success=True, method_used=method_used)
                else:
                    fallback_used = True
                    log("bg_remove_ai", success=False, reason="ai_failed_fallback_auto")
                    result_img, method_used, analysis = remove_bg_smart(img_bytes)
                    method_used = f"{method_used} | fallback from AI"
                    log("bg_remove_auto_fallback", success=True, method_used=method_used, **(analysis or {}))

            elif bg_remove == 'color':
                try:
                    img = Image.open(BytesIO(img_bytes))
                    result_img = remove_bg_color_method(img)
                    method_used = 'color-based (forced)'
                    analysis = {"forced": "color"}
                    log("bg_remove_color", success=True, method_used=method_used)
                except Exception as e:
                    fallback_used = True
                    log("bg_remove_color", success=False, error=str(e), reason="fallback_auto")
                    result_img, method_used, analysis = remove_bg_smart(img_bytes)
                    method_used = f"{method_used} | fallback from color"
                    log("bg_remove_auto_fallback", success=True, method_used=method_used, **(analysis or {}))

        # ---- STEP 3: refine edges ----
        try:
            result_img = refine_edges(result_img, edge_smoothing=1)
            log("refine_edges", success=True, edge_smoothing=1)
        except Exception as e:
            log("refine_edges", success=False, error=str(e))
            raise

        # ---- STEP 4: trim ----
        if do_trim:
            try:
                before = f"{result_img.width}x{result_img.height}"
                result_img = trim_whitespace(result_img)
                after = f"{result_img.width}x{result_img.height}"
                log("trim", success=True, before=before, after=after)
            except Exception as e:
                log("trim", success=False, error=str(e))
                raise
        else:
            log("trim", success=True, skipped=True)

        # ---- encode output ----
        output = BytesIO()
        if output_format == 'webp':
            result_img.save(output, format='WEBP', quality=95, lossless=False)
            mimetype = 'image/webp'
            extension = 'webp'
        else:
            result_img.save(output, format='PNG', optimize=True)
            mimetype = 'image/png'
            extension = 'png'
        output.seek(0)
        log("encode", success=True, format=extension, out_bytes=output.getbuffer().nbytes)

        processing_time = time.time() - start_time
        log("done", success=True, processing_time_s=f"{processing_time:.2f}")

        # ---- send response ----
        response = send_file(
            output,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f'removed_bg.{extension}'
        )

        # Existing informative headers
        response.headers['X-Bg-Remove'] = bg_remove
        response.headers['X-Method-Used'] = method_used
        response.headers['X-Fallback-Used'] = str(fallback_used)

        response.headers['X-Has-Solid-BG'] = str(analysis.get('has_solid_bg', False))
        response.headers['X-Is-Graphic'] = str(analysis.get('is_graphic', False))
        response.headers['X-Already-Transparent'] = str(already_transparent)

        response.headers['X-Enhanced'] = str(enhanced)
        response.headers['X-Enhance-Status'] = enhance_msg

        response.headers['X-Trimmed'] = str(do_trim)
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}s"
        response.headers['X-Output-Size'] = f"{result_img.width}x{result_img.height}"

        # Always attach logs for success too
        return attach_logs_to_response(response)

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log("exception", success=False, error=str(e))

        return json_error({
            "error": "Processing failed",
            "details": str(e),
            "traceback": tb
        }, status=500)




@app.route('/remove-bg/info', methods=['GET'])
def remove_bg_info():
    """Information about the background removal endpoint"""
    return jsonify({
        "endpoint": "/remove-bg",
        "description": "Automatic background removal - works like remove.bg",
        "how_it_works": {
            "step_1": "Analyzes your image automatically",
            "step_2": "Detects if it's a logo/graphic or photo",
            "step_3": "Uses the best method for that image type",
            "step_4": "Returns image with transparent background"
        },
        "supported_inputs": ["PNG", "JPG", "JPEG", "WebP", "BMP", "GIF"],
        "output_formats": ["PNG (default)", "WebP"],
        "usage": {
            "simple": "curl -X POST -F 'image=@your-image.png' http://your-api/remove-bg -o result.png",
            "with_trim": "curl -X POST -F 'image=@your-image.png' -F 'trim=true' http://your-api/remove-bg -o result.png",
            "webp_output": "curl -X POST -F 'image=@your-image.png' -F 'output_format=webp' http://your-api/remove-bg -o result.webp"
        },
        "features": {
            "auto_detection": "Automatically detects logos vs photos",
            "preserves_content": "Keeps all text and elements in logos",
            "smooth_edges": "Anti-aliased edges for professional results",
            "ai_fallback": "Uses AI for complex photo backgrounds"
        }
    })


# ============================================================
# PROCESS-LOGO: Full Pipeline Endpoint
# ============================================================

def enhance_image_fal(image_bytes, wait_timeout=120, poll_interval=1.5):
    """
    fal-ai/seedvr/upscale/image (Queue API)
    - POST returns request_id (not final result)
    - Poll /requests/{id}/status until COMPLETED
    - GET /requests/{id} to fetch output schema (image.url)
    - If sync_mode=True, image.url may be a data URI; decode it.
    """
    if not FAL_KEY:
        return image_bytes, False, "FAL_KEY not configured"

    import time

    def _first(x):
        return x[0] if isinstance(x, list) and x else x

    def _decode_data_uri(data_uri: str) -> bytes:
        # data:<mime>;base64,<data>
        try:
            head, b64 = data_uri.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            return b""

    try:
        # Convert input to base64 data URI (doc allows it)
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        img = Image.open(BytesIO(image_bytes))
        mime_type = "image/png" if (img.format or "").upper() == "PNG" else "image/jpeg"
        data_uri = f"data:{mime_type};base64,{img_base64}"

        headers = {
            "Authorization": f"Key {FAL_KEY}",
            "Content-Type": "application/json",
        }

        # 1) Submit (Queue)
        submit = requests.post(
            "https://queue.fal.run/fal-ai/seedvr/upscale/image",
            headers=headers,
            json={
                "image_url": data_uri,
                # "sync_mode": True,  # optional; if True, output may be data URI
                # You can add other documented fields here (upscale_mode, factor, etc.)
            },
            timeout=60,
        )

        if submit.status_code not in (200, 201, 202):
            return image_bytes, False, f"fal submit failed: HTTP {submit.status_code} {submit.text[:200]}"

        submit_json = _first(submit.json())
        request_id = (submit_json or {}).get("request_id")
        if not request_id:
            return image_bytes, False, "fal submit response missing request_id"

        status_url = f"https://queue.fal.run/fal-ai/seedvr/requests/{request_id}/status"
        result_url = f"https://queue.fal.run/fal-ai/seedvr/requests/{request_id}"

        # 2) Poll status
        start = time.monotonic()
        while (time.monotonic() - start) < wait_timeout:
            st = requests.get(status_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=30)

            if st.status_code not in (200, 202):
                return image_bytes, False, f"fal status failed: HTTP {st.status_code} {st.text[:200]}"

            st_json = _first(st.json()) or {}
            status = st_json.get("status")

            if status in ("COMPLETED", "SUCCEEDED"):
                break
            if status in ("FAILED", "CANCELLED", "ERROR"):
                return image_bytes, False, f"fal job failed: {status}"

            time.sleep(poll_interval)
        else:
            return image_bytes, False, f"fal timeout after {wait_timeout}s"

        # 3) Fetch result (Output Schema: image.url)
        res = requests.get(result_url, headers={"Authorization": f"Key {FAL_KEY}"}, timeout=30)
        if res.status_code != 200:
            return image_bytes, False, f"fal result failed: HTTP {res.status_code} {res.text[:200]}"

        res_json = _first(res.json()) or {}
        image_obj = res_json.get("image") or {}
        out_url = image_obj.get("url")

        if not out_url:
            return image_bytes, False, "fal result missing image.url"

        # 4) Download or decode
        if isinstance(out_url, str) and out_url.startswith("data:"):
            out_bytes = _decode_data_uri(out_url)
            if out_bytes:
                return out_bytes, True, "Enhanced successfully (fal sync_mode data URI)"
            return image_bytes, False, "fal returned data URI but decode failed"

        dl = requests.get(out_url, timeout=60)
        if dl.status_code == 200 and dl.content:
            return dl.content, True, "Enhanced successfully (fal queued)"

        return image_bytes, False, f"fal download failed: HTTP {dl.status_code}"

    except Exception as e:
        return image_bytes, False, f"fal exception: {e}"



def remove_background(img):
    """
    Remove background from image using rembg if available,
    otherwise use simple threshold-based removal
    Returns RGBA image with transparent background
    """
    try:
        from rembg import remove
        # rembg is available, use it
        output = remove(img)
        return output, "rembg"
    except ImportError:
        pass
    
    # Fallback: Simple background removal using corner color detection
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    h, w = data.shape[:2]
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Detect background from corners
    corners = [data[0, 0], data[0, w-1], data[h-1, 0], data[h-1, w-1]]
    corner_rgb = np.array([c[:3] for c in corners], dtype=np.float32)
    corner_std = float(np.std(corner_rgb))
    
    if corner_std < 30:  # Corners are consistent (likely solid background)
        avg_corner = np.mean(corner_rgb, axis=0).astype(np.uint8)
        tolerance = 25
        
        # Create mask for background pixels
        bg_mask = (
            (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
            (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
            (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance)
        )
        
        # Flood fill from corners to get connected background
        bg_u8 = (bg_mask.astype(np.uint8) * 255)
        connected_bg = np.zeros_like(bg_u8)
        
        for sy, sx in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            if bg_u8[sy, sx] > 0:
                temp = bg_u8.copy()
                flood = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(temp, flood, (sx, sy), 128)
                connected_bg[temp == 128] = 255
        
        # Set background pixels to transparent
        data[connected_bg > 0, 3] = 0
        
        return Image.fromarray(data, 'RGBA'), "corner_detection"
    
    return img_rgba, "none"


def trim_whitespace(img):
    """
    Remove transparent/whitespace areas around the logo
    Returns cropped image
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    
    # Find non-transparent pixels
    alpha = data[:, :, 3]
    non_transparent = np.where(alpha > 10)
    
    if len(non_transparent[0]) == 0:
        return img_rgba  # No visible pixels, return as-is
    
    # Get bounding box
    y_min, y_max = non_transparent[0].min(), non_transparent[0].max()
    x_min, x_max = non_transparent[1].min(), non_transparent[1].max()
    
    # Add small padding (2px)
    padding = 2
    y_min = max(0, y_min - padding)
    y_max = min(data.shape[0], y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(data.shape[1], x_max + padding + 1)
    
    # Crop
    cropped = data[y_min:y_max, x_min:x_max]
    
    return Image.fromarray(cropped, 'RGBA')


def determine_logo_type(img):
    """
    Analyze logo to determine if it's black-ish or white-ish
    Returns: "black" or "white"
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Only consider visible pixels (alpha > 10)
    visible_mask = a > 10
    
    if not np.any(visible_mask):
        return "black", 0.0, 0.0  # Default to black if no visible pixels
    
    # Calculate luminance for visible pixels
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    visible_lum = luminance[visible_mask]
    
    # Define thresholds
    dark_threshold = 100
    light_threshold = 200
    
    # Count dark and light pixels
    dark_pixels = np.sum(visible_lum < dark_threshold)
    light_pixels = np.sum(visible_lum > light_threshold)
    total_visible = visible_mask.sum()
    
    dark_ratio = dark_pixels / total_visible if total_visible > 0 else 0
    light_ratio = light_pixels / total_visible if total_visible > 0 else 0
    
    # Determine type based on ratios
    if dark_ratio > light_ratio:
        return "black", dark_ratio, light_ratio
    elif light_ratio > dark_ratio:
        return "white", dark_ratio, light_ratio
    else:
        # If equal or both low, check average luminance
        avg_lum = np.mean(visible_lum)
        if avg_lum < 128:
            return "black", dark_ratio, light_ratio
        else:
            return "white", dark_ratio, light_ratio


def generate_solid_version(img, color):
    """
    Generate solid black or white version of logo
    color: "black" or "white"
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    
    # Only modify visible pixels (alpha > 10)
    visible_mask = data[:, :, 3] > 10
    
    if color == "black":
        data[visible_mask, 0] = 0
        data[visible_mask, 1] = 0
        data[visible_mask, 2] = 0
    else:  # white
        data[visible_mask, 0] = 255
        data[visible_mask, 1] = 255
        data[visible_mask, 2] = 255
    
    return Image.fromarray(data, 'RGBA')


def generate_variant(img, current_type):
    """
    Generate color variant (opposite of current type)
    Uses the same logic as /gen-logo-variant but inline
    """
    img_rgba = img.convert('RGBA')
    data = np.array(img_rgba)
    h, w = data.shape[:2]
    
    r = data[:, :, 0].astype(np.uint8)
    g = data[:, :, 1].astype(np.uint8)
    b = data[:, :, 2].astype(np.uint8)
    a = data[:, :, 3].astype(np.uint8)
    
    # Logo mask (visible pixels)
    alpha_min = 10
    logo_mask = a > alpha_min
    
    if not np.any(logo_mask):
        return img_rgba
    
    # Thresholds
    dark_thr = 100
    white_cut = 220
    
    # Detect dark and white pixels in logo
    blackish = logo_mask & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
    whiteish = logo_mask & (r > white_cut) & (g > white_cut) & (b > white_cut)
    
    dark_px = int(np.sum(blackish))
    white_px = int(np.sum(whiteish))
    logo_px = int(np.sum(logo_mask))
    
    dark_ratio = dark_px / max(1, logo_px)
    white_ratio = white_px / max(1, logo_px)
    
    changed = False
    
    # Apply transformation based on detected type
    if current_type == "black" and dark_ratio > 0.1:
        # Convert dark to white
        data[blackish, 0] = 255
        data[blackish, 1] = 255
        data[blackish, 2] = 255
        changed = True
    elif current_type == "white" and white_ratio > 0.1:
        # Convert white to black
        data[whiteish, 0] = 0
        data[whiteish, 1] = 0
        data[whiteish, 2] = 0
        changed = True
    
    # If no significant change, invert logo colors
    if not changed:
        data[logo_mask, 0] = 255 - data[logo_mask, 0]
        data[logo_mask, 1] = 255 - data[logo_mask, 1]
        data[logo_mask, 2] = 255 - data[logo_mask, 2]
    
    return Image.fromarray(data, 'RGBA')


def image_to_base64(img):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _ensure_ftp_dir(ftp: ftplib.FTP, path: str):
    """
    Ensure `path` exists on the FTP server and cwd into it.
    Creates missing segments one by one.
    Supports paths like: /variant or variant or /a/b/c
    """
    if not path:
        return

    # Normalize to segments
    path = path.strip()
    is_abs = path.startswith("/")
    parts = [p for p in path.strip("/").split("/") if p]

    # Go to root for absolute paths (best effort)
    if is_abs:
        try:
            ftp.cwd("/")
        except ftplib.error_perm:
            # Some FTP servers don't allow cwd "/" explicitly; ignore.
            pass

    for part in parts:
        try:
            ftp.cwd(part)
        except ftplib.error_perm:
            # Create then enter
            ftp.mkd(part)
            ftp.cwd(part)

def upload_images_to_ftp(images_dict, folder_id: str):
    if not all([FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL]):
        return None, "FTP not configured (set FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL)"

    urls = {}
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=30)
        ftp.login(FTP_USER, FTP_PASS)

        # Ensure base dir exists and cd into it
        _ensure_ftp_dir(ftp, FTP_DIR)

        # Ensure folder_id exists and cd into it
        _ensure_ftp_dir(ftp, folder_id)

        base_url = FTP_BASE_URL.rstrip("/")

        for filename, pil_img in images_dict.items():
            img_buffer = BytesIO()
            pil_img.save(img_buffer, format="PNG", optimize=True)
            img_buffer.seek(0)

            ftp.storbinary(f"STOR {filename}", img_buffer)

            # Public URL = base_url + /folder_id/filename
            # (FTP_DIR is not appended here because FTP_BASE_URL should already represent the public base)
            urls[filename] = f"{base_url}/{folder_id}/{filename}"

        ftp.quit()
        return urls, "success"

    except Exception as e:
        try:
            ftp.quit()
        except Exception:
            pass
        return None, str(e)


def get_folder_id_from_request():
    """
    Accept folder_id from:
      1) multipart/form-data: folder_id
      2) query param: ?folder_id=
      3) header: X-Folder-Id
    Falls back to uuid4 if missing.

    Sanitizes to FTP-safe: a-z A-Z 0-9 _ -
    """
    raw = (
        (request.form.get("folder_id") if request.form else None)
        or request.args.get("folder_id")
        or request.headers.get("X-Folder-Id")
        or ""
    ).strip()

    if not raw:
        return str(uuid.uuid4())

    # Replace spaces with dashes and remove unsafe chars
    safe = raw.replace(" ", "-")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", safe)

    # Avoid empty / weird edge cases after sanitize
    if not safe:
        return str(uuid.uuid4())

    # Optional: cap length for sanity
    return safe[:80]

@app.route('/process-logo', methods=['POST'])
def process_logo():
    """
    LOGO PROCESSING PIPELINE (v3) - With FTP Upload
    
    Accepts: image (file) - required
    
    Pipeline:
    1. Load image (expects transparent background already)
    2. Determine logo type (black-ish or white-ish)
    3. Generate 2 versions:
       - original_{type}: The original
       - original_{opposite}: Color variant (inverted)
    4. Create unique folder and upload both to FTP
    
    Returns: JSON with folder_id and image URLs
    
    Required Environment Variables for FTP:
        - FTP_HOST: FTP server hostname
        - FTP_USER: FTP username
        - FTP_PASS: FTP password
        - FTP_DIR: Base remote directory (default: /logos)
        - FTP_BASE_URL: Public URL base (e.g., https://cdn.example.com)
    """
    
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Check FTP configuration
        if not all([FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL]):
            return jsonify({
                "error": "FTP not configured",
                "message": "Please set FTP_HOST, FTP_USER, FTP_PASS, and FTP_BASE_URL environment variables"
            }), 500
        
        # Read image bytes
        image_bytes = file.read()
        
        # ✅ folder_id from request (instead of always generating uuid)
        folder_id = get_folder_id_from_request()
        
        # Track processing steps
        processing_log = []
        
        # ============================================================
        # STEP 1: Load image
        # ============================================================
        img = Image.open(BytesIO(image_bytes))
        img = img.convert('RGBA')
        
        processing_log.append({
            "step": "load_image",
            "success": True,
            "size": f"{img.width}x{img.height}"
        })
        
        # ============================================================
        # STEP 2: Determine logo type
        # ============================================================
        logo_type, dark_ratio, light_ratio = determine_logo_type(img)
        opposite_type = "white" if logo_type == "black" else "black"
        
        processing_log.append({
            "step": "analyze",
            "detected_type": logo_type,
            "dark_ratio": f"{dark_ratio:.4f}",
            "light_ratio": f"{light_ratio:.4f}"
        })
        
        # ============================================================
        # STEP 3: Generate 2 versions
        # ============================================================
        
        # Original
        original_key = f"original_{logo_type}"
        original_img = img.copy()
        
        # Variant (opposite color)
        variant_key = f"original_{opposite_type}"
        variant_img = generate_variant(img, logo_type)
        
        processing_log.append({
            "step": "generate_versions",
            "success": True,
            "versions": [original_key, variant_key]
        })
        
        # ============================================================
        # STEP 4: Upload to FTP
        # ============================================================
        images_to_upload = {
            f"{original_key}.png": original_img,
            f"{variant_key}.png": variant_img
        }
        
        urls, ftp_status = upload_images_to_ftp(images_to_upload, folder_id)
        
        if urls is None:
            return jsonify({
                "error": "FTP upload failed",
                "message": ftp_status,
                "processing_log": processing_log
            }), 500
        
        processing_log.append({
            "step": "ftp_upload",
            "success": True,
            "folder_id": folder_id,
            "files_uploaded": len(urls)
        })
        
        # ============================================================
        # OUTPUT: JSON with URLs
        # ============================================================
        return jsonify({
            "success": True,
            "folder_id": folder_id,
            "detected_type": logo_type,
            "processing_log": processing_log,
            "images": {
                original_key: {
                    "description": f"Original logo ({logo_type})",
                    "url": urls[f"{original_key}.png"]
                },
                variant_key: {
                    "description": f"Color variant ({opposite_type})",
                    "url": urls[f"{variant_key}.png"]
                }
            }
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/check-dependencies', methods=['GET'])
def check_dependencies():
    """Check if all dependencies are available"""
    deps = {}
    
    try:
        import flask
        deps['flask'] = flask.__version__
    except:
        deps['flask'] = 'NOT INSTALLED'
    
    try:
        import PIL
        deps['pillow'] = PIL.__version__
    except:
        deps['pillow'] = 'NOT INSTALLED'
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except:
        deps['numpy'] = 'NOT INSTALLED'
    
    try:
        import cv2
        deps['opencv'] = cv2.__version__
    except Exception as e:
        deps['opencv'] = f'NOT INSTALLED: {str(e)}'
    
    try:
        from rembg import remove
        deps['rembg'] = 'INSTALLED'
    except ImportError:
        deps['rembg'] = 'NOT INSTALLED (will use fallback bg removal)'
    
    return jsonify({
        "dependencies": deps,
        "opencv_available": 'cv2' in dir(),
        "fal_api_configured": FAL_KEY is not None,
        "ftp_configured": all([FTP_HOST, FTP_USER, FTP_PASS, FTP_BASE_URL])
    })

@app.route('/admin', methods=['GET'])
def admin():
    """Fun admin endpoint for keep-alive pings"""
    import random
    from datetime import datetime
    
    funny_messages = [
        "🎨 Image processor standing by, captain!",
        "🖼️ Still here, converting pixels like a boss!",
        "🎭 Awake and ready to make colors dance!",
        "🚀 Service online! No images harmed in the making of this response.",
        "🎪 The pixel circus is open for business!"
    ]
    
    return jsonify({
        "status": "alive_and_kicking",
        "message": random.choice(funny_messages),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "coffee_level": f"{random.randint(60, 100)}%"
    })

@app.route('/smart-logo-variant', methods=['POST'])
def smart_logo_variant():
    """
    SMART LOGO VARIANT (v1)
    Always returns a *different* usable variant, while protecting gradients.

    Priority rule (your requirement):
    1) If dark-ish pixels are > 30% of solid logo area:
         -> ONLY convert dark-ish => white-ish (layered palette starting from pure white)
         -> do NOT invert / swap

    Otherwise:
    2) If no dark-ish but there is white-ish:
         -> convert white-ish => black-ish (layered)
    3) If tiny dark-ish + lots of white-ish (e.g., black 'T' in white circle):
         -> swap: dark-ish => white-ish AND white-ish => black-ish
    4) If nothing changes:
         -> outline-only fallback so client never gets identical original

    Notes:
    - Gradient components are detected per connected-component and skipped.
    - “Layer separation” is preserved by quantizing target pixels to 2..4 levels and mapping to a palette.
    """

    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # -----------------------
        # Params (safe defaults)
        # -----------------------
        dark_thr = int(request.form.get('threshold', DEFAULT_THRESHOLD))  # same as your /replace-dark-to-white
        dark_thr = max(0, min(255, dark_thr))

        white_threshold = int(request.form.get('white_threshold', 35))
        white_threshold = max(1, min(80, white_threshold))
        white_cut = 255 - white_threshold

        alpha_min = int(request.form.get('alpha_min', 10))
        alpha_min = max(0, min(255, alpha_min))

        # Ratios
        heavy_dark_cut = float(request.form.get('heavy_dark_ratio', 0.30))      # ✅ your priority trigger
        heavy_dark_cut = max(0.0, min(1.0, heavy_dark_cut))

        small_dark_ratio = float(request.form.get('small_dark_ratio', 0.02))    # “tiny black”
        small_dark_ratio = max(0.0, min(1.0, small_dark_ratio))

        high_white_ratio = float(request.form.get('high_white_ratio', 0.40))    # “mostly white”
        high_white_ratio = max(0.0, min(1.0, high_white_ratio))

        gradient_mode = request.form.get('gradient_mode', 'skip').lower().strip()
        # skip = do not recolor gradients (recommended)
        if gradient_mode not in ['skip', 'preserve', 'allow']:
            gradient_mode = 'skip'

        # -----------------------
        # Load image
        # -----------------------
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        h, w = data.shape[:2]

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)

        original = data.copy()

        # ============================================================
        # STEP 1: Background detection (reused pattern from your code)
        # ============================================================
        corners = [data[0, 0], data[0, w - 1], data[h - 1, 0], data[h - 1, w - 1]]
        corner_rgb = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_a   = np.array([c[3]  for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_rgb, axis=0)
        avg_alpha  = float(np.mean(corner_a))
        corner_std = float(np.std(corner_rgb))
        corners_consistent = corner_std < 30

        bg_mask = (a <= alpha_min)

        if avg_alpha > 200 and corners_consistent:
            mean_corner = float(np.mean(avg_corner))
            tolerance = 20

            if mean_corner > 240:
                bg_mask = bg_mask | ((r > 250) & (g > 250) & (b > 250) & (a > 200))
            elif mean_corner < 15:
                bg_mask = bg_mask | ((r < 5) & (g < 5) & (b < 5) & (a > 200))
            else:
                bg_mask = bg_mask | (
                    (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                    (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                    (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                    (a > 200)
                )

            # Flood fill from corners to keep only corner-connected background
            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for sy, sx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                if potential_bg[sy, sx] > 0:
                    temp = potential_bg.copy()
                    flood = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(temp, flood, (sx, sy), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        logo_mask = (~bg_mask) & (a > alpha_min)
        if int(np.sum(logo_mask)) == 0:
            logo_mask = (a > alpha_min)
            bg_mask = ~logo_mask

        # ============================================================
        # STEP 2: Connected components on logo pixels
        # ============================================================
        mask_u8 = (logo_mask.astype(np.uint8) * 255)
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        # Luminance for decisions
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        # ============================================================
        # STEP 3: Gradient detection per component (ported from your hybrid idea)
        # ============================================================
        def is_gradient_component(comp_bool: np.ndarray) -> bool:
            if gradient_mode == 'allow':
                return False
            if gradient_mode in ['skip', 'preserve']:
                # detect gradient (skip recolor if True)
                pass

            k = np.ones((3, 3), np.uint8)
            interior = cv2.erode(comp_bool.astype(np.uint8), k, iterations=1).astype(bool)
            if interior.sum() < 80:
                interior = comp_bool

            lum_vals = luminance[interior].astype(np.float32)
            if lum_vals.size < 140:
                return False

            hist = np.histogram(lum_vals, bins=32)[0].astype(np.float32)
            p = hist / (hist.sum() + 1e-9)
            entropy = float(-np.sum(p * np.log(p + 1e-9)))
            norm_entropy = entropy / float(np.log(len(p)))
            dom_mass = float(np.sort(p)[-3:].sum())
            occupancy = int(np.sum(hist > 0))

            ys, xs = np.where(interior)
            X = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones(xs.size, np.float32)])
            yv = lum_vals
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((yv - pred) ** 2))
            ss_tot = float(np.sum((yv - float(yv.mean())) ** 2)) + 1e-9
            r2 = 1.0 - (ss_res / ss_tot)

            lum_std = float(lum_vals.std())
            lum_rng = float(lum_vals.max() - lum_vals.min())

            entropy_gradient = (norm_entropy > 0.70 and dom_mass < 0.55 and occupancy > 10 and lum_rng > 20)
            linear_gradient  = (r2 > 0.60 and lum_std > 8 and lum_rng > 20)
            return bool(entropy_gradient or linear_gradient)

        # ============================================================
        # STEP 4: Solid-only stats to decide global mode
        # ============================================================
        solid_logo_mask = np.zeros((h, w), dtype=bool)

        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue
            comp = (cc_labels == lab) & logo_mask
            if comp.sum() == 0:
                continue
            if gradient_mode in ['skip', 'preserve'] and is_gradient_component(comp):
                # skip gradient pixels from stats (so decisions reflect solid only)
                continue
            solid_logo_mask |= comp

        solid_px = int(np.sum(solid_logo_mask))
        if solid_px == 0:
            # If everything is gradient or too small, fallback to outline-only
            solid_logo_mask = logo_mask
            solid_px = int(np.sum(solid_logo_mask))

        blackish_solid = solid_logo_mask & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
        whiteish_solid = solid_logo_mask & (r > white_cut) & (g > white_cut) & (b > white_cut)

        dark_px  = int(np.sum(blackish_solid))
        white_px = int(np.sum(whiteish_solid))

        dark_ratio  = dark_px  / max(1, solid_px)
        white_ratio = white_px / max(1, solid_px)

        # ✅ Priority: heavy dark => simple dark->white only
        mode = None
        if dark_ratio >= heavy_dark_cut:
            mode = "heavy-dark-to-white"
        else:
            # Tiny dark + lots of white => swap (fix “T in white circle”)
            if (dark_ratio <= small_dark_ratio) and (white_ratio >= high_white_ratio) and (dark_px > 0) and (white_px > 0):
                mode = "swap-bw"
            else:
                # No dark => make a variant by turning white-ish into black-ish
                if dark_px == 0 and white_px > 0:
                    mode = "white-to-black"
                else:
                    # Default: dark -> white (main goal)
                    if dark_px > 0:
                        mode = "dark-to-white"
                    else:
                        mode = "outline-only"

        # ============================================================
        # STEP 5: Layered palette mapping (prevents borders merging)
        # ============================================================
        def apply_layered_palette(target_mask: np.ndarray, to: str):
            """
            Quantize luminance in target_mask into 2..4 levels and map to palette.
            to: 'white' or 'black'
            """
            ys, xs = np.where(target_mask)
            if ys.size == 0:
                return 0

            lum = luminance[ys, xs].astype(np.float32)

            # Decide K (2..4) based on luminance range & pixel count
            lum_rng = float(lum.max() - lum.min()) if lum.size else 0.0
            if lum.size < 600 or lum_rng < 18:
                K = 2
            elif lum_rng < 60:
                K = 3
            else:
                K = 4

            if lum.size < K:
                K = 2

            # KMeans on 1D luminance for "layer" bins
            Z = lum.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.4)
            _, labels_k, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

            # Order centers from darkest -> lightest
            order = np.argsort(centers.flatten())
            lut = np.empty(K, dtype=np.int32)
            for rank, old in enumerate(order):
                lut[int(old)] = rank
            ranks = lut[labels_k.flatten()]  # 0=darkest, K-1=lightest

            # Palettes (start from pure white / pure black) preserving separation
            if to == 'white':
                # darkest becomes pure white, lighter layers become slightly darker whites
                # (keeps borders/shadows visible)
                palette = [255, 235, 215, 195][:K]
            else:
                # whitest becomes pure black, other layers become lighter blacks
                palette = [0, 40, 80, 120][:K]

            out = np.take(np.array(palette, dtype=np.uint8), ranks)

            data[ys, xs, 0] = out
            data[ys, xs, 1] = out
            data[ys, xs, 2] = out

            return int(ys.size)

        # ============================================================
        # STEP 6: Apply transform per component (skip gradients)
        # ============================================================
        changed_pixels = 0
        gradients_skipped = 0

        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue

            comp = (cc_labels == lab) & logo_mask
            if int(np.sum(comp)) == 0:
                continue

            comp_is_grad = False
            if gradient_mode in ['skip', 'preserve'] and is_gradient_component(comp):
                comp_is_grad = True

            if comp_is_grad:
                gradients_skipped += 1
                continue

            # Within this SOLID component, compute blackish/whiteish
            blackish = comp & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
            whiteish = comp & (r > white_cut) & (g > white_cut) & (b > white_cut)

            if mode == "heavy-dark-to-white":
                changed_pixels += apply_layered_palette(blackish, to='white')

            elif mode == "dark-to-white":
                changed_pixels += apply_layered_palette(blackish, to='white')

            elif mode == "white-to-black":
                changed_pixels += apply_layered_palette(whiteish, to='black')

            elif mode == "swap-bw":
                # Swap both, layered
                changed_pixels += apply_layered_palette(blackish, to='white')
                changed_pixels += apply_layered_palette(whiteish, to='black')

            else:
                # outline-only handled later
                pass

        # ============================================================
        # STEP 7: Outline-only fallback (guarantees a different file)
        # ============================================================
        # If nothing changed OR mode was outline-only, draw a thin edge stroke
        # around logo alpha silhouette.
        def add_outline():
            nonlocal changed_pixels

            alpha_mask = (a > alpha_min).astype(np.uint8) * 255
            if alpha_mask.sum() == 0:
                return

            k = np.ones((3, 3), np.uint8)
            dil = cv2.dilate(alpha_mask, k, iterations=1)
            ero = cv2.erode(alpha_mask, k, iterations=1)
            edge = (dil > 0) & (ero == 0)

            # Decide outline color by average luminance of logo pixels
            logo_lum = luminance[logo_mask]
            avg_lum = float(np.mean(logo_lum)) if logo_lum.size else 128.0
            outline_is_dark = (avg_lum > 140)  # bright logo -> dark outline

            if outline_is_dark:
                c = np.array([0, 0, 0], dtype=np.uint8)
            else:
                c = np.array([255, 255, 255], dtype=np.uint8)

            before = data[edge, 0:3].copy()
            data[edge, 0:3] = c
            data[edge, 3] = np.maximum(data[edge, 3], 220).astype(np.uint8)

            # count changed edge pixels
            changed_pixels += int(np.sum(np.any(before != c, axis=1))) if before.size else int(edge.sum())

        if mode == "outline-only":
            add_outline()
        else:
            # If result equals original (or no pixels changed), enforce outline
            if changed_pixels == 0 or np.array_equal(data, original):
                mode = "outline-only"
                add_outline()

        # ============================================================
        # OUTPUT
        # ============================================================
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='smart_logo_variant.png'
        )

        response.headers['X-Variant-Mode'] = mode
        response.headers['X-Dark-Ratio'] = f"{dark_ratio:.4f}"
        response.headers['X-White-Ratio'] = f"{white_ratio:.4f}"
        response.headers['X-Changed-Pixels'] = str(int(changed_pixels))
        response.headers['X-Gradients-Skipped'] = str(int(gradients_skipped))
        response.headers['X-Dark-Threshold'] = str(dark_thr)
        response.headers['X-White-Threshold'] = str(white_threshold)
        response.headers['X-Heavy-Dark-Cut'] = str(heavy_dark_cut)

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/gen-logo-variant', methods=['POST'])
def gen_logo_variant():
    """
    GEN LOGO VARIANT (v3) - Full Pipeline + Autonomous Color Variant
    
    Only accepts: image (file)
    
    Pipeline:
    1. Enhance image (fal.ai SeedVR Upscale) - if FAL_KEY configured
    2. Remove background (rembg or fallback)
    3. Trim whitespace
    4. Generate color variant (auto-decided):
       - If dark-ish => convert to white-ish
       - If white-ish => convert to black-ish
       - If neither => invert logo colors
    
    Returns: PNG image file (the variant)
    """

    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read image bytes
        image_bytes = file.read()
        
        # Track processing
        enhance_status = "skipped"
        bg_removal_method = "none"
        
        # ============================================================
        # STEP 1: Enhance image (if fal.ai configured)
        # ============================================================
        if FAL_KEY:
            image_bytes, enhanced, enhance_msg = enhance_image_fal(image_bytes)
            enhance_status = "success" if enhanced else f"failed: {enhance_msg}"
        
        # Load as PIL Image
        img = Image.open(BytesIO(image_bytes))
        
        # ============================================================
        # STEP 2: Remove background
        # ============================================================
        # Check if already has transparency
        has_transparency = False
        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            has_transparency = np.any(alpha < 255)
        
        if has_transparency:
            img = img.convert('RGBA')
            bg_removal_method = "already_transparent"
        else:
            img, bg_removal_method = remove_background(img)
        
        # ============================================================
        # STEP 3: Trim whitespace
        # ============================================================
        img = trim_whitespace(img)
        
        # ============================================================
        # STEP 4: Generate color variant (existing logic)
        # ============================================================
        data = np.array(img)
        h, w = data.shape[:2]

        # Auto-configured params
        dark_thr = 100
        white_cut = 220
        alpha_min = 10
        heavy_dark_cut = 0.30
        small_dark_ratio = 0.02
        high_white_ratio = 0.40

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)

        original = data.copy()

        # Background detection
        corners = [data[0, 0], data[0, w - 1], data[h - 1, 0], data[h - 1, w - 1]]
        corner_rgb = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_a   = np.array([c[3]  for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_rgb, axis=0)
        avg_alpha  = float(np.mean(corner_a))
        corner_std = float(np.std(corner_rgb))
        corners_consistent = corner_std < 30

        bg_mask = (a <= alpha_min)

        if avg_alpha > 200 and corners_consistent:
            mean_corner = float(np.mean(avg_corner))
            tolerance = 20

            if mean_corner > 240:
                bg_mask = bg_mask | ((r > 250) & (g > 250) & (b > 250) & (a > 200))
            elif mean_corner < 15:
                bg_mask = bg_mask | ((r < 5) & (g < 5) & (b < 5) & (a > 200))
            else:
                bg_mask = bg_mask | (
                    (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                    (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                    (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                    (a > 200)
                )

            # Flood fill from corners
            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for sy, sx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                if potential_bg[sy, sx] > 0:
                    temp = potential_bg.copy()
                    flood = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(temp, flood, (sx, sy), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        logo_mask = (~bg_mask) & (a > alpha_min)
        if int(np.sum(logo_mask)) == 0:
            logo_mask = (a > alpha_min)
            bg_mask = ~logo_mask

        # Connected components
        mask_u8 = (logo_mask.astype(np.uint8) * 255)
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        # Luminance
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        # Gradient detection
        def is_gradient_component(comp_bool: np.ndarray) -> bool:
            k = np.ones((3, 3), np.uint8)
            interior = cv2.erode(comp_bool.astype(np.uint8), k, iterations=1).astype(bool)
            if interior.sum() < 80:
                interior = comp_bool

            lum_vals = luminance[interior].astype(np.float32)
            if lum_vals.size < 140:
                return False

            hist = np.histogram(lum_vals, bins=32)[0].astype(np.float32)
            p = hist / (hist.sum() + 1e-9)
            entropy = float(-np.sum(p * np.log(p + 1e-9)))
            norm_entropy = entropy / float(np.log(len(p)))
            dom_mass = float(np.sort(p)[-3:].sum())
            occupancy = int(np.sum(hist > 0))

            ys, xs = np.where(interior)
            X = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones(xs.size, np.float32)])
            yv = lum_vals
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((yv - pred) ** 2))
            ss_tot = float(np.sum((yv - float(yv.mean())) ** 2)) + 1e-9
            r2 = 1.0 - (ss_res / ss_tot)

            lum_std = float(lum_vals.std())
            lum_rng = float(lum_vals.max() - lum_vals.min())

            entropy_gradient = (norm_entropy > 0.70 and dom_mass < 0.55 and occupancy > 10 and lum_rng > 20)
            linear_gradient  = (r2 > 0.60 and lum_std > 8 and lum_rng > 20)
            return bool(entropy_gradient or linear_gradient)

        # Solid-only stats
        solid_logo_mask = np.zeros((h, w), dtype=bool)

        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue
            comp = (cc_labels == lab) & logo_mask
            if comp.sum() == 0:
                continue
            if is_gradient_component(comp):
                continue
            solid_logo_mask |= comp

        solid_px = int(np.sum(solid_logo_mask))
        if solid_px == 0:
            solid_logo_mask = logo_mask
            solid_px = int(np.sum(solid_logo_mask))

        blackish_solid = solid_logo_mask & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
        whiteish_solid = solid_logo_mask & (r > white_cut) & (g > white_cut) & (b > white_cut)

        dark_px  = int(np.sum(blackish_solid))
        white_px = int(np.sum(whiteish_solid))

        dark_ratio  = dark_px  / max(1, solid_px)
        white_ratio = white_px / max(1, solid_px)

        # Determine mode
        mode = None
        if dark_ratio >= heavy_dark_cut:
            mode = "heavy-dark-to-white"
        else:
            if (dark_ratio <= small_dark_ratio) and (white_ratio >= high_white_ratio) and (dark_px > 0) and (white_px > 0):
                mode = "swap-bw"
            else:
                if dark_px == 0 and white_px > 0:
                    mode = "white-to-black"
                else:
                    if dark_px > 0:
                        mode = "dark-to-white"
                    else:
                        mode = "invert-logo"

        # Layered palette mapping
        def apply_layered_palette(target_mask: np.ndarray, to: str):
            ys, xs = np.where(target_mask)
            if ys.size == 0:
                return 0

            lum = luminance[ys, xs].astype(np.float32)
            lum_rng = float(lum.max() - lum.min()) if lum.size else 0.0
            
            if lum.size < 600 or lum_rng < 18:
                K = 2
            elif lum_rng < 60:
                K = 3
            else:
                K = 4

            if lum.size < K:
                K = 2

            Z = lum.reshape(-1, 1).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.4)
            _, labels_k, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

            order = np.argsort(centers.flatten())
            lut = np.empty(K, dtype=np.int32)
            for rank, old in enumerate(order):
                lut[int(old)] = rank
            ranks = lut[labels_k.flatten()]

            if to == 'white':
                palette = [255, 235, 215, 195][:K]
            else:
                palette = [0, 40, 80, 120][:K]

            out = np.take(np.array(palette, dtype=np.uint8), ranks)

            data[ys, xs, 0] = out
            data[ys, xs, 1] = out
            data[ys, xs, 2] = out

            return int(ys.size)

        # Apply transform
        changed_pixels = 0
        gradients_skipped = 0

        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue

            comp = (cc_labels == lab) & logo_mask
            if int(np.sum(comp)) == 0:
                continue

            if is_gradient_component(comp):
                gradients_skipped += 1
                continue

            blackish = comp & (r < dark_thr) & (g < dark_thr) & (b < dark_thr)
            whiteish = comp & (r > white_cut) & (g > white_cut) & (b > white_cut)

            if mode == "heavy-dark-to-white":
                changed_pixels += apply_layered_palette(blackish, to='white')
            elif mode == "dark-to-white":
                changed_pixels += apply_layered_palette(blackish, to='white')
            elif mode == "white-to-black":
                changed_pixels += apply_layered_palette(whiteish, to='black')
            elif mode == "swap-bw":
                changed_pixels += apply_layered_palette(blackish, to='white')
                changed_pixels += apply_layered_palette(whiteish, to='black')

        # Invert fallback
        def invert_logo_colors():
            nonlocal changed_pixels
            logo_pixels = logo_mask & (a > alpha_min)
            if not np.any(logo_pixels):
                return
            data[logo_pixels, 0] = 255 - data[logo_pixels, 0]
            data[logo_pixels, 1] = 255 - data[logo_pixels, 1]
            data[logo_pixels, 2] = 255 - data[logo_pixels, 2]
            changed_pixels += int(np.sum(logo_pixels))

        if mode == "invert-logo":
            invert_logo_colors()
        else:
            if changed_pixels == 0 or np.array_equal(data, original):
                mode = "invert-logo"
                invert_logo_colors()

        # ============================================================
        # OUTPUT
        # ============================================================
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='gen_logo_variant.png'
        )

        response.headers['X-Enhance-Status'] = enhance_status
        response.headers['X-BG-Removal-Method'] = bg_removal_method
        response.headers['X-Variant-Mode'] = mode
        response.headers['X-Dark-Ratio'] = f"{dark_ratio:.4f}"
        response.headers['X-White-Ratio'] = f"{white_ratio:.4f}"
        response.headers['X-Changed-Pixels'] = str(int(changed_pixels))
        response.headers['X-Gradients-Skipped'] = str(int(gradients_skipped))

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/logo-type', methods=['POST'])
def logo_type():
    """
    LOGO TYPE DETECTION
    
    Analyzes a logo and returns whether it's "black" or "white"
    
    Pipeline:
    1. Remove background (if needed)
    2. Trim whitespace
    3. Analyze pixel luminance
    4. Return type as plain text
    
    Returns: Plain text "black" or "white"
    """
    
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Load image
        img = Image.open(file.stream)
        
        # Remove background if needed
        has_transparency = False
        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            has_transparency = np.any(alpha < 255)
        
        if not has_transparency:
            img, _ = remove_background(img)
        else:
            img = img.convert('RGBA')
        
        # Trim whitespace
        img = trim_whitespace(img)
        
        # Determine type
        logo_type_result, _, _ = determine_logo_type(img)
        
        # Return plain text
        return logo_type_result, 200, {'Content-Type': 'text/plain'}
    
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500


@app.route('/smart-print-ready', methods=['POST'])
def smart_print_ready():
    """
    SMART PRINT-READY (Hybrid v7)
    - Converts logo to print-ready grayscale (white or black scheme) with transparency preserved.
    - Handles mixed logos: per connected-component decide Gradient vs Stepped Layers.
    - Keeps your “group/layer” behavior for solid regions while avoiding ugly banding for gradients.

    Params (form-data):
    - image: file (required)
    - print_color: 'white' | 'black' (required)
    - layers: 'auto' | 2..6 (default: 'auto')         # used for STEPPED components
    - white_step: 5..30 (default: 10)                 # used for STEPPED components
    - black_step: 15..50 (default: 33)                # used for STEPPED components
    - gradient_mode: 'auto' | 'smooth' | 'stepped' (default: 'auto')
        auto   = detect per component
        smooth = force all components smooth
        stepped= force all components stepped
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        print_color = request.form.get('print_color', '').lower().strip()
        num_layers = request.form.get('layers', 'auto')
        white_step = int(request.form.get('white_step', 10))
        black_step = int(request.form.get('black_step', 33))
        gradient_mode = request.form.get('gradient_mode', 'auto').lower().strip()

        if print_color not in ['white', 'black']:
            return jsonify({
                "error": "print_color is required",
                "usage": "print_color=white (for dark shirts) or print_color=black (for light shirts)"
            }), 400

        white_step = max(5, min(30, white_step))
        black_step = max(15, min(50, black_step))
        if gradient_mode not in ['auto', 'smooth', 'stepped']:
            gradient_mode = 'auto'

        # ---------------------------
        # Load image
        # ---------------------------
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)
        original_alpha = a.copy()

        # ============================================================
        # STEP 1: BACKGROUND DETECTION (your logic, kept)
        # ============================================================
        corners = [
            data[0, 0],
            data[0, width - 1],
            data[height - 1, 0],
            data[height - 1, width - 1]
        ]

        corner_colors = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_alphas = np.array([c[3] for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_colors, axis=0)
        avg_alpha = float(np.mean(corner_alphas))
        corner_std = float(np.std(corner_colors))

        corners_consistent = corner_std < 30

        if avg_alpha < 128:
            bg_type = "transparent"
            bg_mask = original_alpha < 10
        elif corners_consistent and float(np.mean(avg_corner)) > 240:
            bg_type = "white"
            bg_mask = (r > 250) & (g > 250) & (b > 250) & (original_alpha > 200)
        elif corners_consistent and float(np.mean(avg_corner)) < 15:
            bg_type = "black"
            bg_mask = (r < 5) & (g < 5) & (b < 5) & (original_alpha > 200)
        elif corners_consistent:
            bg_type = "colored"
            tolerance = 20
            bg_mask = (
                (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                (original_alpha > 200)
            )
        else:
            bg_type = "mixed/none"
            bg_mask = original_alpha < 10

        # Always treat near-transparent as background
        bg_mask = bg_mask | (original_alpha < 10)

        # Flood fill from corners for non-transparent backgrounds
        if bg_type not in ["transparent", "mixed/none"]:
            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for start_y, start_x in [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]:
                if potential_bg[start_y, start_x] > 0:
                    temp = potential_bg.copy()
                    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
                    cv2.floodFill(temp, flood_mask, (start_x, start_y), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        # Logo mask (non-background, but keep alpha relevance)
        logo_mask = (~bg_mask) & (original_alpha > 10)

        if int(np.sum(logo_mask)) == 0:
            # fallback: alpha-only
            logo_mask = original_alpha > 10
            bg_mask = ~logo_mask
            bg_type = "fallback-transparent-only"

        # ============================================================
        # STEP 2: LUMINANCE
        # ============================================================
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        logo_luminance = luminance[logo_mask]
        if logo_luminance.size == 0:
            return jsonify({"error": "No logo pixels found"}), 400

        lum_min = float(np.min(logo_luminance))
        lum_max = float(np.max(logo_luminance))
        lum_range = float(lum_max - lum_min)

        # ============================================================
        # STEP 3+4: HYBRID PER CONNECTED COMPONENT
        # ============================================================
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[bg_mask] = [0, 0, 0, 0]

        # Label connected components on logo pixels
        mask_u8 = (logo_mask.astype(np.uint8) * 255)
        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        def _parse_layers(val):
            if isinstance(val, str) and val.lower() == 'auto':
                return 'auto'
            try:
                return max(2, min(6, int(val)))
            except:
                return 'auto'

        layers_setting = _parse_layers(num_layers)

        # --- gradient detection per component ---
        def is_gradient_component(comp_bool: np.ndarray) -> bool:
            # Respect explicit mode
            if gradient_mode == 'smooth':
                return True
            if gradient_mode == 'stepped':
                return False

            # Remove edges (anti-alias) to avoid false detection
            k = np.ones((3, 3), np.uint8)
            interior = cv2.erode(comp_bool.astype(np.uint8), k, iterations=1).astype(bool)
            if interior.sum() < 80:
                interior = comp_bool  # fallback if too thin

            lum_vals = luminance[interior].astype(np.float32)
            if lum_vals.size < 140:
                return False

            # Histogram entropy (smooth gradients -> higher entropy, less dominance)
            hist = np.histogram(lum_vals, bins=32)[0].astype(np.float32)
            p = hist / (hist.sum() + 1e-9)
            entropy = float(-np.sum(p * np.log(p + 1e-9)))
            norm_entropy = entropy / float(np.log(len(p)))
            dom_mass = float(np.sort(p)[-3:].sum())     # top-3 bin mass
            occupancy = int(np.sum(hist > 0))

            # Linear fit: lum ≈ ax + by + c (gradients often fit well)
            ys, xs = np.where(interior)
            X = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), np.ones(xs.size, np.float32)])
            yv = lum_vals
            coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((yv - pred) ** 2))
            ss_tot = float(np.sum((yv - float(yv.mean())) ** 2)) + 1e-9
            r2 = 1.0 - (ss_res / ss_tot)

            lum_std = float(lum_vals.std())
            lum_rng = float(lum_vals.max() - lum_vals.min())

            # Decision rules:
            entropy_gradient = (norm_entropy > 0.70 and dom_mass < 0.55 and occupancy > 10 and lum_rng > 20)
            linear_gradient = (r2 > 0.60 and lum_std > 8 and lum_rng > 20)

            return bool(entropy_gradient or linear_gradient)

        # Smooth gradient output range (more “solid” by default)
        # If you want even MORE solid gradients: white 255..235, black 0..25
        if print_color == 'white':
            grad_out_max, grad_out_min = 255, 200
        else:
            grad_out_min, grad_out_max = 0, 70

        stepped_components = 0
        gradient_components = 0
        used_gray_values = set()

        # Process each component
        for lab in range(1, num_cc):
            area = int(cc_stats[lab, cv2.CC_STAT_AREA])
            if area < 25:
                continue

            comp = (cc_labels == lab) & logo_mask
            if int(np.sum(comp)) == 0:
                continue

            component_is_grad = is_gradient_component(comp)
            if component_is_grad:
                gradient_components += 1
            else:
                stepped_components += 1

            ys, xs = np.where(comp)
            comp_lum = luminance[ys, xs].astype(np.float32)
            comp_alpha = original_alpha[ys, xs].astype(np.uint8)

            if component_is_grad:
                # -------- Smooth mapping (vectorized) --------
                if lum_range > 1e-6:
                    norm = (comp_lum - lum_min) / lum_range
                else:
                    norm = np.zeros_like(comp_lum)

                # Optional gentle gamma to reduce harsh contrast
                norm = np.clip(norm, 0.0, 1.0) ** 1.1

                if print_color == 'white':
                    out = grad_out_max - norm * (grad_out_max - grad_out_min)
                else:
                    out = grad_out_min + norm * (grad_out_max - grad_out_min)

                out = np.clip(out, 0, 255).astype(np.uint8)

                result[ys, xs, 0] = out
                result[ys, xs, 1] = out
                result[ys, xs, 2] = out
                result[ys, xs, 3] = comp_alpha

                # track used grays (sampled, to avoid huge sets)
                if out.size > 0:
                    used_gray_values.update(np.unique(out[::max(1, out.size // 5000)]).tolist())

            else:
                # -------- Stepped layers (per component kmeans) --------
                comp_min = float(comp_lum.min())
                comp_max = float(comp_lum.max())
                comp_rng = float(comp_max - comp_min)

                # Decide layers for this component
                if layers_setting == 'auto':
                    if comp_rng < 30:
                        L = 2
                    elif comp_rng < 80:
                        L = 3
                    elif comp_rng < 150:
                        L = 4
                    else:
                        L = 5
                else:
                    L = int(layers_setting)

                # If too flat, keep as single tone
                if comp_rng < 10 or comp_lum.size < L:
                    L = 1
                    ranks = np.zeros(comp_lum.size, dtype=np.int32)
                else:
                    Z = comp_lum.reshape(-1, 1).astype(np.float32)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                    _, labels_k, centers = cv2.kmeans(Z, L, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

                    order = np.argsort(centers.flatten())  # darkest -> lightest
                    lut = np.empty(L, dtype=np.int32)
                    for rank, old in enumerate(order):
                        lut[int(old)] = rank
                    ranks = lut[labels_k.flatten()]

                # Build output gray palette for this component
                if print_color == 'white':
                    grays = np.empty(L, dtype=np.uint8)
                    for rank in range(L):
                        if rank == 0:
                            grays[rank] = 255
                        else:
                            darkness = min(rank * white_step, 50)
                            v = int(255 * (100 - darkness) / 100)
                            grays[rank] = max(128, v)
                else:
                    grays = np.empty(L, dtype=np.uint8)
                    for rank in range(L):
                        if rank == 0:
                            grays[rank] = 0
                        else:
                            lightness = min(rank * black_step, 85)
                            v = int(255 * lightness / 100)
                            grays[rank] = min(220, v)

                out = grays[np.clip(ranks, 0, L - 1)].astype(np.uint8)

                result[ys, xs, 0] = out
                result[ys, xs, 1] = out
                result[ys, xs, 2] = out
                result[ys, xs, 3] = comp_alpha

                used_gray_values.update(grays.tolist())

        # Extra safety fallback
        if int(np.sum(result[:, :, 3] > 0)) == 0:
            ys, xs = np.where(original_alpha > 10)
            base = 255 if print_color == 'white' else 0
            result[ys, xs, 0] = base
            result[ys, xs, 1] = base
            result[ys, xs, 2] = base
            result[ys, xs, 3] = original_alpha[ys, xs]
            bg_type = bg_type + "|fallback-filled"

        # ============================================================
        # STEP 5: OUTPUT
        # ============================================================
        result_img = Image.fromarray(result, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        used_step = white_step if print_color == 'white' else black_step

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'print_ready_{print_color}.png'
        )

        # Headers for debugging/telemetry
        response.headers['X-Print-Color'] = print_color.upper()
        response.headers['X-Background-Type'] = bg_type
        response.headers['X-Processing-Method'] = "hybrid-components"
        response.headers['X-Gradient-Mode'] = gradient_mode
        response.headers['X-Gradient-Components'] = f"{gradient_components}/{max(1, (gradient_components + stepped_components))}"
        response.headers['X-Stepped-Components'] = str(stepped_components)
        response.headers['X-Step'] = f"{used_step}%"
        response.headers['X-Luminance-Range'] = f"{lum_min:.0f}-{lum_max:.0f}"

        # Keep this short so headers don’t explode
        if used_gray_values:
            sample = sorted(list(used_gray_values))
            if len(sample) > 40:
                sample = sample[:20] + ["..."] + sample[-20:]
            response.headers['X-Used-Grays'] = str(sample)

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/smart-print-ready/analyze', methods=['POST'])
def smart_print_ready_analyze():
    """
    Analyze logo - detect gradients, layers, and expected output
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        height, width = data.shape[:2]
        
        r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
        
        # Background detection
        corners = [data[0,0], data[0,-1], data[-1,0], data[-1,-1]]
        corner_colors = np.array([c[:3] for c in corners])
        avg_corner = np.mean(corner_colors, axis=0)
        avg_alpha = np.mean([c[3] for c in corners])
        corner_std = np.std(corner_colors)
        
        if avg_alpha < 128:
            bg_type = "transparent"
        elif corner_std < 30 and np.mean(avg_corner) > 240:
            bg_type = "white"
        elif corner_std < 30 and np.mean(avg_corner) < 15:
            bg_type = "black"
        else:
            bg_type = "other"
        
        # Luminance analysis
        luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        logo_mask = a > 10
        
        if not np.any(logo_mask):
            return jsonify({"error": "No logo pixels found"}), 400
        
        logo_lum = luminance[logo_mask]
        lum_min, lum_max = float(np.min(logo_lum)), float(np.max(logo_lum))
        lum_range = lum_max - lum_min
        
        # Gradient detection
        sobel_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        logo_gradient = gradient_magnitude[logo_mask]
        avg_gradient = float(np.mean(logo_gradient))
        
        # Histogram analysis
        hist, _ = np.histogram(logo_lum, bins=32)
        hist_normalized = hist / np.max(hist)
        peaks = int(np.sum(hist_normalized > 0.15))
        hist_spread = int(np.sum(hist_normalized > 0.05))
        
        # Gradient detection logic
        has_gradient = (hist_spread > 20 and peaks < 4) or \
                       (avg_gradient > 15 and peaks < 5) or \
                       (lum_range > 100 and peaks < 4)
        
        # Recommendations
        if has_gradient:
            recommended_mode = "smooth"
            reason = "Gradient detected - smooth mapping recommended"
        else:
            recommended_mode = "stepped"
            reason = f"Solid colors detected - {peaks} distinct layers found"
        
        return jsonify({
            "image": {
                "dimensions": f"{width}x{height}",
                "logo_pixels": int(np.sum(logo_mask))
            },
            "background": {
                "type": bg_type
            },
            "luminance": {
                "min": lum_min,
                "max": lum_max,
                "range": lum_range
            },
            "gradient_analysis": {
                "average_gradient_magnitude": avg_gradient,
                "histogram_peaks": peaks,
                "histogram_spread": hist_spread,
                "has_gradient": has_gradient
            },
            "recommendation": {
                "gradient_mode": recommended_mode,
                "reason": reason,
                "suggested_layers": "smooth" if has_gradient else max(2, min(peaks, 5))
            },
            "api_usage": {
                "for_gradient": "POST /smart-print-ready with gradient_mode=smooth",
                "for_solid": "POST /smart-print-ready with gradient_mode=stepped",
                "auto_detect": "POST /smart-print-ready with gradient_mode=auto (default)"
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Analysis failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@app.route('/force-solid-with-outline', methods=['POST'])
def force_solid_with_outline():
    """
    Convert logo to solid color while keeping it readable
    Adds outline/stroke to preserve shape definition
    
    Parameters:
    - image: file (required)
    - fill_color: 'black' or 'white' (default: 'black')
    - outline_color: 'white' or 'black' (default: opposite of fill)
    - outline_width: integer pixels (default: 3)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Get parameters
        fill_color = request.form.get('fill_color', 'black').lower()
        outline_color = request.form.get('outline_color', None)
        outline_width = int(request.form.get('outline_width', 3))
        
        # Auto-set outline color to contrast with fill
        if outline_color is None:
            outline_color = 'white' if fill_color == 'black' else 'black'
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        
        # Get original alpha channel (shape of logo)
        original_alpha = img.split()[3]
        
        # Create solid color version
        if fill_color == 'white':
            solid_fill = Image.new('RGB', img.size, (255, 255, 255))
        else:
            solid_fill = Image.new('RGB', img.size, (0, 0, 0))
        
        # Create outline by dilating the alpha channel
        # This makes the shape slightly bigger
        alpha_array = np.array(original_alpha)
        
        # Use morphological dilation to create outline
        kernel = np.ones((outline_width * 2 + 1, outline_width * 2 + 1), np.uint8)
        dilated = cv2.dilate(alpha_array, kernel, iterations=1)
        
        # The outline is the difference between dilated and original
        outline_mask = (dilated > 0) & (alpha_array == 0)
        
        # Create outline image
        if outline_color == 'white':
            outline_img = Image.new('RGBA', img.size, (255, 255, 255, 0))
        else:
            outline_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        
        outline_data = np.array(outline_img)
        if outline_color == 'white':
            outline_data[outline_mask] = [255, 255, 255, 255]
        else:
            outline_data[outline_mask] = [0, 0, 0, 255]
        
        outline_img = Image.fromarray(outline_data, 'RGBA')
        
        # Composite: outline first, then solid fill on top
        result = Image.new('RGBA', img.size, (0, 0, 0, 0))
        result.paste(outline_img, (0, 0), outline_img)
        result.paste(solid_fill, (0, 0), original_alpha)
        
        # Save
        output = BytesIO()
        result.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'solid_{fill_color}_with_outline.png'
        )
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/force-solid-readable', methods=['POST'])
def force_solid_readable():
    """
    Convert logo to solid color while keeping ALL details readable
    Uses luminance-based edge preservation for better results
    
    Parameters:
    - image: file (required)
    - base_color: 'black' or 'white' (default: 'white')
    - edge_strength: 1-10 (default: 3, higher=thicker edges)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        base_color = request.form.get('base_color', 'white').lower()
        edge_strength = int(request.form.get('edge_strength', 3))
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb = np.array(img.convert('RGB'))
        alpha = np.array(img.split()[3])
        
        # Convert to grayscale to detect luminance differences
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Use Sobel edge detection (better for internal lines)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        sobel_edges = (sobel_edges / sobel_edges.max() * 255).astype(np.uint8)
        _, edge_mask = cv2.threshold(sobel_edges, 20, 255, cv2.THRESH_BINARY)
        
        # Thicken edges based on strength parameter
        if edge_strength > 1:
            kernel = np.ones((edge_strength, edge_strength), np.uint8)
            edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
        
        # Create result
        result = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        # Set base color for all non-transparent pixels
        non_transparent = alpha > 0
        
        if base_color == 'white':
            result[non_transparent, 0:3] = [255, 255, 255]  # White base
            # Make edges black
            edge_pixels = (edge_mask > 0) & non_transparent
            result[edge_pixels, 0:3] = [0, 0, 0]
        else:
            result[non_transparent, 0:3] = [0, 0, 0]  # Black base
            # Make edges white
            edge_pixels = (edge_mask > 0) & non_transparent
            result[edge_pixels, 0:3] = [255, 255, 255]
        
        # Apply original alpha
        result[:, :, 3] = alpha
        
        # Create final image
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        # Debug info
        edge_pixel_count = np.sum(edge_pixels)
        total_opaque = np.sum(non_transparent)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'readable_{base_color}.png'
        )
        
        response.headers['X-Edge-Pixels'] = str(edge_pixel_count)
        response.headers['X-Total-Opaque-Pixels'] = str(total_opaque)
        response.headers['X-Edge-Percentage'] = f"{(edge_pixel_count/total_opaque*100):.2f}%"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/smart-solid-conversion', methods=['POST'])
def smart_solid_conversion():
    """
    Intelligently convert multi-color logo to solid color
    Darker colors become strokes, lighter colors become fill
    
    Parameters:
    - image: file (required)
    - output_color: 'black' or 'white' (default: 'white')
    - invert_logic: 'true' or 'false' (default: 'false')
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        output_color = request.form.get('output_color', 'white').lower()
        invert_logic = request.form.get('invert_logic', 'false').lower() == 'true'
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb = np.array(img.convert('RGB'))
        alpha = np.array(img.split()[3])
        
        # Calculate luminance (brightness) for each pixel
        luminance = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
        
        # Non-transparent pixels only
        non_transparent = alpha > 0
        
        # Find threshold - use mean luminance of non-transparent pixels
        if np.any(non_transparent):
            threshold = np.mean(luminance[non_transparent])
        else:
            threshold = 127
        
        # Classify pixels as "dark" (strokes) or "light" (fill)
        if invert_logic:
            is_stroke = (luminance > threshold) & non_transparent
        else:
            is_stroke = (luminance < threshold) & non_transparent
        
        # Create result
        result = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        if output_color == 'white':
            # White fill, black strokes
            result[non_transparent, 0:3] = [255, 255, 255]
            result[is_stroke, 0:3] = [0, 0, 0]
        else:
            # Black fill, white strokes
            result[non_transparent, 0:3] = [0, 0, 0]
            result[is_stroke, 0:3] = [255, 255, 255]
        
        # Apply alpha
        result[:, :, 3] = alpha
        
        result_img = Image.fromarray(result, 'RGBA')
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        stroke_pixels = np.sum(is_stroke)
        fill_pixels = np.sum(non_transparent) - stroke_pixels
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_{output_color}.png'
        )
        
        response.headers['X-Stroke-Pixels'] = str(stroke_pixels)
        response.headers['X-Fill-Pixels'] = str(fill_pixels)
        response.headers['X-Luminance-Threshold'] = f"{threshold:.2f}"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500
        
@app.route('/force-solid-black', methods=['POST'])
def force_solid_black():
    """
    Convert ANY logo to solid black while PRESERVING transparency
    Ultra-safe version that guarantees transparent pixels stay transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Open image - force RGBA mode
        img = Image.open(file.stream)
        
        # Ensure we have an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the data as numpy array
        data = np.array(img, dtype=np.uint8)
        
        # Split into separate channels
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]
        alpha = data[:, :, 3]
        
        # Create new array with same shape
        result = np.zeros_like(data)
        
        # Set RGB to black for all pixels
        result[:, :, 0] = 0  # R
        result[:, :, 1] = 0  # G
        result[:, :, 2] = 0  # B
        
        # CRITICAL: Copy original alpha channel exactly as-is
        result[:, :, 3] = alpha
        
        # Create image from array
        result_img = Image.fromarray(result, mode='RGBA')
        
        # Save as PNG with transparency
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=False)
        output.seek(0)
        
        # Stats
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha == 0)
        semi_transparent = np.sum((alpha > 0) & (alpha < 255))
        fully_opaque = np.sum(alpha == 255)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='solid_black.png'
        )
        
        response.headers['X-Total-Pixels'] = str(total_pixels)
        response.headers['X-Transparent-Pixels'] = str(transparent_pixels)
        response.headers['X-Semi-Transparent-Pixels'] = str(semi_transparent)
        response.headers['X-Opaque-Pixels'] = str(fully_opaque)
        response.headers['X-Output-Color'] = 'BLACK'
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/force-solid-white', methods=['POST'])
def force_solid_white():
    """
    Convert ANY logo to solid white while PRESERVING transparency
    Ultra-safe version that guarantees transparent pixels stay transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Open image - force RGBA mode
        img = Image.open(file.stream)
        
        # Ensure we have an alpha channel
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the data as numpy array
        data = np.array(img, dtype=np.uint8)
        
        # Split into separate channels
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]
        alpha = data[:, :, 3]
        
        # Create new array with same shape
        result = np.zeros_like(data)
        
        # Set RGB to white for all pixels
        result[:, :, 0] = 255  # R
        result[:, :, 1] = 255  # G
        result[:, :, 2] = 255  # B
        
        # CRITICAL: Copy original alpha channel exactly as-is
        result[:, :, 3] = alpha
        
        # Create image from array
        result_img = Image.fromarray(result, mode='RGBA')
        
        # Save as PNG with transparency
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=False)
        output.seek(0)
        
        # Stats
        total_pixels = alpha.size
        transparent_pixels = np.sum(alpha == 0)
        semi_transparent = np.sum((alpha > 0) & (alpha < 255))
        fully_opaque = np.sum(alpha == 255)
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='solid_white.png'
        )
        
        response.headers['X-Total-Pixels'] = str(total_pixels)
        response.headers['X-Transparent-Pixels'] = str(transparent_pixels)
        response.headers['X-Semi-Transparent-Pixels'] = str(semi_transparent)
        response.headers['X-Opaque-Pixels'] = str(fully_opaque)
        response.headers['X-Output-Color'] = 'WHITE'
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/check-transparency', methods=['POST'])
def check_transparency():
    """Check transparency information of uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream)
        
        original_mode = img.mode
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        data = np.array(img)
        alpha = data[:, :, 3]
        
        return jsonify({
            "original_mode": original_mode,
            "converted_mode": "RGBA",
            "dimensions": f"{img.width}x{img.height}",
            "total_pixels": int(alpha.size),
            "transparency_info": {
                "fully_transparent": int(np.sum(alpha == 0)),
                "semi_transparent": int(np.sum((alpha > 0) & (alpha < 255))),
                "fully_opaque": int(np.sum(alpha == 255))
            },
            "has_transparency": bool(np.any(alpha < 255)),
            "min_alpha": int(np.min(alpha)),
            "max_alpha": int(np.max(alpha)),
            "avg_alpha": float(np.mean(alpha))
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/debug-smart-color', methods=['POST'])
def debug_smart_color():
    """Debug version - shows why smart-color-replace fails"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        target_type = request.form.get('target_type', 'dark').lower()
        
        img = Image.open(file.stream).convert('RGBA')
        rgb = img.convert('RGB')
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Test each condition
        kernel_size = 5
        h_var = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_var = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_var = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        total_var = h_var + s_var + v_var
        
        solid_mask = total_var < 100
        colored_mask = s > 30
        
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        target_mask = solid_mask & colored_mask & brightness_mask
        
        total_pixels = h.size
        
        return jsonify({
            "image_size": f"{rgb_array.shape[1]}x{rgb_array.shape[0]}",
            "total_pixels": int(total_pixels),
            "results": {
                "solid_pixels": int(np.sum(solid_mask)),
                "solid_percentage": f"{(np.sum(solid_mask)/total_pixels)*100:.2f}%",
                "colored_pixels": int(np.sum(colored_mask)),
                "colored_percentage": f"{(np.sum(colored_mask)/total_pixels)*100:.2f}%",
                "brightness_matched": int(np.sum(brightness_mask)),
                "brightness_percentage": f"{(np.sum(brightness_mask)/total_pixels)*100:.2f}%",
                "final_matched": int(np.sum(target_mask)),
                "final_percentage": f"{(np.sum(target_mask)/total_pixels)*100:.2f}%"
            },
            "diagnosis": "PASS" if np.sum(target_mask) > 0 else "FAIL - No pixels meet all criteria",
            "likely_reason": "Image has gradients (not solid colors)" if np.sum(solid_mask) < (total_pixels * 0.1) else "Check brightness_type parameter"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/replace-dark-to-white', methods=['POST'])
def replace_dark():
    """
    Replace dark-ish pixels with white — with an AUTO invert mode:

    ✅ If the logo area is basically ONLY black/white (2-tone),
       then SWAP them:
         - black-ish  -> white
         - white-ish  -> black
       (anti-aliased mid pixels are pushed to nearest side)

    ✅ If the logo area contains other colors/tones (multi-tone),
       then ONLY:
         - black-ish -> white
       (white + other colors remain unchanged)

    Form-data params:
    - image: file (required)
    - threshold: 0..255 (default DEFAULT_THRESHOLD)  # used for "black-ish"
    Optional:
    - white_threshold: 1..80 (default 35)           # used for "white-ish"
    - bw_ratio: 0..1 (default 0.97)                 # how “pure” B/W the logo must be to invert
    - alpha_min: 0..255 (default 10)                # pixels below are treated as transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # ---------- params ----------
        threshold = int(request.form.get('threshold', DEFAULT_THRESHOLD))
        if threshold < 0 or threshold > 255:
            return jsonify({"error": "Threshold must be between 0 and 255"}), 400

        white_threshold = int(request.form.get('white_threshold', 35))
        white_threshold = max(1, min(80, white_threshold))

        bw_ratio_cut = float(request.form.get('bw_ratio', 0.97))
        bw_ratio_cut = max(0.0, min(1.0, bw_ratio_cut))

        alpha_min = int(request.form.get('alpha_min', 10))
        alpha_min = max(0, min(255, alpha_min))

        # ---------- load ----------
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        h, w = data.shape[:2]

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)

        # ============================================================
        # STEP 1: BACKGROUND DETECTION (so we don’t invert the background)
        # ============================================================
        corners = [
            data[0, 0],
            data[0, w - 1],
            data[h - 1, 0],
            data[h - 1, w - 1]
        ]
        corner_rgb = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_a = np.array([c[3] for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_rgb, axis=0)
        avg_alpha = float(np.mean(corner_a))
        corner_std = float(np.std(corner_rgb))
        corners_consistent = corner_std < 30

        # Default: treat transparent as background
        bg_mask = (a <= alpha_min)

        if avg_alpha > 200 and corners_consistent:
            mean_corner = float(np.mean(avg_corner))
            tolerance = 20

            # white-ish corner bg
            if mean_corner > 240:
                bg_mask = bg_mask | ((r > 250) & (g > 250) & (b > 250) & (a > 200))
            # black-ish corner bg
            elif mean_corner < 15:
                bg_mask = bg_mask | ((r < 5) & (g < 5) & (b < 5) & (a > 200))
            else:
                # colored-ish bg (use corner color with tolerance)
                bg_mask = bg_mask | (
                    (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                    (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                    (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                    (a > 200)
                )

            # Flood fill from corners (keeps only corner-connected bg)
            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for sy, sx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                if potential_bg[sy, sx] > 0:
                    temp = potential_bg.copy()
                    flood = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(temp, flood, (sx, sy), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        logo_mask = (~bg_mask) & (a > alpha_min)
        if int(np.sum(logo_mask)) == 0:
            # fallback: alpha-only logo
            logo_mask = (a > alpha_min)
            bg_mask = ~logo_mask

        # ============================================================
        # STEP 2: Decide if logo is “mostly BW only”
        # ============================================================
        white_cut = 255 - white_threshold

        blackish = logo_mask & (r < threshold) & (g < threshold) & (b < threshold)
        whiteish = logo_mask & (r > white_cut) & (g > white_cut) & (b > white_cut)
        bw_mask = blackish | whiteish

        logo_px = int(np.sum(logo_mask))
        bw_px = int(np.sum(bw_mask))

        # Ratio of pixels that are either black-ish or white-ish
        bw_ratio = (bw_px / max(1, logo_px))

        # Two-tone if almost everything is BW (allow tiny anti-aliasing mid pixels)
        is_two_tone_bw = (bw_ratio >= bw_ratio_cut)

        # Luminance for anti-aliased pixels handling (only when in invert mode)
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        # ============================================================
        # STEP 3: Apply transformation
        # ============================================================
        if is_two_tone_bw:
            # Invert black-ish and white-ish inside logo area
            data[blackish, 0] = 255
            data[blackish, 1] = 255
            data[blackish, 2] = 255

            data[whiteish, 0] = 0
            data[whiteish, 1] = 0
            data[whiteish, 2] = 0

            # Any “mid” pixels (usually anti-aliasing) -> push to nearest side
            mid = logo_mask & (~blackish) & (~whiteish)
            if int(np.sum(mid)) > 0:
                mid_to_white = mid & (luminance < 128)   # darker edge becomes white after invert
                mid_to_black = mid & (~mid_to_white)

                data[mid_to_white, 0:3] = [255, 255, 255]
                data[mid_to_black, 0:3] = [0, 0, 0]
        else:
            # Multi-tone: ONLY make dark-ish parts white (your current behavior)
            dark_mask = blackish
            data[dark_mask, 0:3] = [255, 255, 255]

        # ============================================================
        # OUTPUT
        # ============================================================
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_image.png'
        )

        # Helpful debug headers
        response.headers['X-Mode'] = 'invert-bw' if is_two_tone_bw else 'dark-to-white'
        response.headers['X-BW-Ratio'] = f"{bw_ratio:.4f}"
        response.headers['X-Threshold'] = str(threshold)
        response.headers['X-White-Threshold'] = str(white_threshold)

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/replace-light-to-dark', methods=['POST'])
def replace_light():
    """
    Replace light/white-ish pixels with black — with an AUTO invert mode:

    ✅ If the logo area is basically ONLY black/white (2-tone),
       then SWAP them:
         - white-ish -> black
         - black-ish -> white
       (anti-aliased mid pixels are pushed to nearest side)

    ✅ If the logo area contains other colors/tones (multi-tone),
       then ONLY:
         - white-ish -> black
       (dark + other colors remain unchanged)

    Form-data params:
    - image: file (required)
    - threshold: 0..255 (default 200)          # used for "white-ish" (r/g/b > threshold)
    Optional:
    - black_threshold: 1..120 (default 80)     # used for "black-ish" (r/g/b < black_threshold)
    - bw_ratio: 0..1 (default 0.97)            # how “pure” B/W the logo must be to invert
    - alpha_min: 0..255 (default 10)           # pixels below are treated as transparent
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error

    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # ---------- params ----------
        threshold = int(request.form.get('threshold', 200))
        threshold = max(0, min(255, threshold))

        black_threshold = int(request.form.get('black_threshold', 80))
        black_threshold = max(1, min(120, black_threshold))

        bw_ratio_cut = float(request.form.get('bw_ratio', 0.97))
        bw_ratio_cut = max(0.0, min(1.0, bw_ratio_cut))

        alpha_min = int(request.form.get('alpha_min', 10))
        alpha_min = max(0, min(255, alpha_min))

        # ---------- load ----------
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        h, w = data.shape[:2]

        r = data[:, :, 0].astype(np.uint8)
        g = data[:, :, 1].astype(np.uint8)
        b = data[:, :, 2].astype(np.uint8)
        a = data[:, :, 3].astype(np.uint8)

        # ============================================================
        # STEP 1: BACKGROUND DETECTION (so we don’t invert the background)
        # ============================================================
        corners = [
            data[0, 0],
            data[0, w - 1],
            data[h - 1, 0],
            data[h - 1, w - 1]
        ]
        corner_rgb = np.array([c[:3] for c in corners], dtype=np.float32)
        corner_a = np.array([c[3] for c in corners], dtype=np.float32)

        avg_corner = np.mean(corner_rgb, axis=0)
        avg_alpha = float(np.mean(corner_a))
        corner_std = float(np.std(corner_rgb))
        corners_consistent = corner_std < 30

        bg_mask = (a <= alpha_min)

        if avg_alpha > 200 and corners_consistent:
            mean_corner = float(np.mean(avg_corner))
            tolerance = 20

            if mean_corner > 240:
                bg_mask = bg_mask | ((r > 250) & (g > 250) & (b > 250) & (a > 200))
            elif mean_corner < 15:
                bg_mask = bg_mask | ((r < 5) & (g < 5) & (b < 5) & (a > 200))
            else:
                bg_mask = bg_mask | (
                    (np.abs(r.astype(np.int16) - int(avg_corner[0])) < tolerance) &
                    (np.abs(g.astype(np.int16) - int(avg_corner[1])) < tolerance) &
                    (np.abs(b.astype(np.int16) - int(avg_corner[2])) < tolerance) &
                    (a > 200)
                )

            potential_bg = (bg_mask.astype(np.uint8) * 255)
            connected_bg = np.zeros_like(potential_bg)

            for sy, sx in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                if potential_bg[sy, sx] > 0:
                    temp = potential_bg.copy()
                    flood = np.zeros((h + 2, w + 2), np.uint8)
                    cv2.floodFill(temp, flood, (sx, sy), 128)
                    connected_bg[temp == 128] = 255

            bg_mask = connected_bg > 0

        logo_mask = (~bg_mask) & (a > alpha_min)
        if int(np.sum(logo_mask)) == 0:
            logo_mask = (a > alpha_min)
            bg_mask = ~logo_mask

        # ============================================================
        # STEP 2: Decide if logo is “mostly BW only”
        # ============================================================
        whiteish = logo_mask & (r > threshold) & (g > threshold) & (b > threshold)
        blackish = logo_mask & (r < black_threshold) & (g < black_threshold) & (b < black_threshold)
        bw_mask = whiteish | blackish

        logo_px = int(np.sum(logo_mask))
        bw_px = int(np.sum(bw_mask))
        bw_ratio = (bw_px / max(1, logo_px))

        is_two_tone_bw = (bw_ratio >= bw_ratio_cut)

        # Luminance for pushing anti-aliased mid pixels
        luminance = (0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32))

        # ============================================================
        # STEP 3: Apply transformation
        # ============================================================
        if is_two_tone_bw:
            # Swap inside logo
            data[whiteish, 0:3] = [0, 0, 0]
            data[blackish, 0:3] = [255, 255, 255]

            # Push mid pixels to nearest side
            mid = logo_mask & (~whiteish) & (~blackish)
            if int(np.sum(mid)) > 0:
                # After invert: lighter edges become white, darker edges become black
                mid_to_black = mid & (luminance >= 128)
                mid_to_white = mid & (~mid_to_black)

                data[mid_to_black, 0:3] = [0, 0, 0]
                data[mid_to_white, 0:3] = [255, 255, 255]
        else:
            # Multi-tone: only white-ish -> black (your original intention)
            data[whiteish, 0:3] = [0, 0, 0]

        # ============================================================
        # OUTPUT
        # ============================================================
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG', optimize=True)
        output.seek(0)

        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='light_to_dark.png'
        )

        response.headers['X-Mode'] = 'invert-bw' if is_two_tone_bw else 'light-to-dark'
        response.headers['X-BW-Ratio'] = f"{bw_ratio:.4f}"
        response.headers['X-Threshold'] = str(threshold)
        response.headers['X-Black-Threshold'] = str(black_threshold)

        return response

    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/replace-to-color', methods=['POST'])
def replace_to_color():
    """
    Replace dark or light colors with any target color
    RECOMMENDED: This is the most reliable endpoint
    
    Parameters:
    - image: file (required)
    - target_hue: 0-360 (required) - color to change TO
    - brightness_type: 'dark' or 'light' (default: 'dark')
    - threshold: 0-255 (default: 100 for dark, 200 for light)
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Get parameters
        target_hue = int(request.form.get('target_hue', 0))
        brightness_type = request.form.get('brightness_type', 'dark').lower()
        
        if brightness_type == 'light':
            threshold = int(request.form.get('threshold', 200))
        else:
            threshold = int(request.form.get('threshold', 100))
        
        # Open image
        img = Image.open(file.stream).convert('RGBA')
        rgb_img = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        
        # Convert to numpy array
        rgb_array = np.array(rgb_img)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Create mask based on brightness
        if brightness_type == 'light':
            # Find light pixels
            brightness_mask = v > threshold
        else:
            # Find dark pixels
            brightness_mask = v < threshold
        
        # Check if any pixels match
        if not np.any(brightness_mask):
            return jsonify({
                "error": f"No {brightness_type} pixels found with threshold {threshold}",
                "suggestion": f"Try adjusting threshold or use brightness_type='{('light' if brightness_type == 'dark' else 'dark')}'"
            }), 400
        
        # Convert target hue to OpenCV range (0-180)
        target_hue_cv = (target_hue % 360) / 2
        
        # Replace hue for matching pixels
        h[brightness_mask] = target_hue_cv
        
        # Boost saturation to make color visible (at least 100)
        s[brightness_mask] = np.maximum(s[brightness_mask], 100)
        
        # Rebuild HSV
        hsv[:,:,0] = h
        hsv[:,:,1] = s
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        # Reapply alpha channel
        if alpha:
            result_img.putalpha(alpha)
        
        # Save and return
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        pixels_changed = np.sum(brightness_mask)
        total_pixels = v.size
        percentage = (pixels_changed / total_pixels) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'{brightness_type}_to_hue{target_hue}.png'
        )
        
        response.headers['X-Pixels-Changed'] = str(pixels_changed)
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        response.headers['X-Target-Hue'] = str(target_hue)
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/smart-color-replace', methods=['POST'])
def smart_color_replace():
    """
    Auto-detect dominant solid color and replace it
    Fixed version with better saturation handling
    """
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        new_hue = int(request.form.get('new_hue', 0))
        target_type = request.form.get('target_type', 'dark').lower()
        
        # Optional: allow custom saturation threshold
        min_saturation = int(request.form.get('min_saturation', 10))  # Lowered from 30
        
        img = Image.open(file.stream).convert('RGBA')
        rgb = img.convert('RGB')
        alpha = img.split()[3] if img.mode == 'RGBA' else None
        rgb_array = np.array(rgb)
        
        # Convert to HSV
        hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Calculate variance to find solid colors
        kernel_size = 5
        h_var = cv2.blur(h**2, (kernel_size, kernel_size)) - cv2.blur(h, (kernel_size, kernel_size))**2
        s_var = cv2.blur(s**2, (kernel_size, kernel_size)) - cv2.blur(s, (kernel_size, kernel_size))**2
        v_var = cv2.blur(v**2, (kernel_size, kernel_size)) - cv2.blur(v, (kernel_size, kernel_size))**2
        
        total_var = h_var + s_var + v_var
        
        # Masks
        solid_mask = total_var < 100
        colored_mask = s > min_saturation  # FIXED: Lower threshold
        
        if target_type == 'light':
            brightness_mask = v > 150
        else:
            brightness_mask = v < 120
        
        target_mask = solid_mask & colored_mask & brightness_mask
        
        # If still no match, try without solid requirement
        if not np.any(target_mask):
            target_mask = colored_mask & brightness_mask
            
            if not np.any(target_mask):
                # Last resort: ignore saturation requirement for very desaturated images
                target_mask = brightness_mask
                
                if not np.any(target_mask):
                    return jsonify({
                        "error": f"No {target_type} pixels found",
                        "suggestion": f"Try target_type='{('light' if target_type == 'dark' else 'dark')}'"
                    }), 400
        
        # Find dominant hue
        valid_hues = h[target_mask]
        hist, bin_edges = np.histogram(valid_hues, bins=36, range=(0, 180))
        dominant_bin = np.argmax(hist)
        dominant_hue = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2
        
        # Match similar hues (more lenient tolerance)
        hue_diff = np.abs(h - dominant_hue)
        hue_diff = np.minimum(hue_diff, 180 - hue_diff)
        hue_match = hue_diff < 40  # Increased from 20
        
        final_mask = target_mask & hue_match
        
        # If still no match after hue filtering, just use brightness mask
        if not np.any(final_mask):
            final_mask = target_mask
        
        # Replace hue
        hsv[:,:,0][final_mask] = new_hue / 2
        
        # Boost saturation to make color visible
        hsv[:,:,1][final_mask] = np.maximum(hsv[:,:,1][final_mask], 100)
        
        # Convert back
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result_img = Image.fromarray(result_rgb, 'RGB')
        
        if alpha:
            result_img.putalpha(alpha)
        
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        pixels_changed = np.sum(final_mask)
        percentage = (pixels_changed / h.size) * 100
        
        response = send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'smart_replaced_{target_type}.png'
        )
        
        response.headers['X-Detected-Hue'] = str(int(dominant_hue * 2))
        response.headers['X-Pixels-Changed'] = str(int(pixels_changed))
        response.headers['X-Change-Percentage'] = f"{percentage:.2f}%"
        
        return response
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": "Processing failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/invert-colors', methods=['POST'])
def invert_colors():
    """Invert all colors (create negative)"""
    auth_error = verify_api_key()
    if auth_error:
        return auth_error
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGBA')
        data = np.array(img)
        
        # Invert RGB, keep alpha
        data[:, :, 0] = 255 - data[:, :, 0]
        data[:, :, 1] = 255 - data[:, :, 1]
        data[:, :, 2] = 255 - data[:, :, 2]
        
        result_img = Image.fromarray(data, 'RGBA')
        output = BytesIO()
        result_img.save(output, format='PNG')
        output.seek(0)
        
        return send_file(
            output,
            mimetype='image/png',
            as_attachment=True,
            download_name='inverted.png'
        )
        
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG)