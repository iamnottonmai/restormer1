import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ─── Restormer Architecture (matches checkpoint key names exactly) ────────────

class LayerNorm(nn.Module):
    """Channel-first LayerNorm. Checkpoint key: norm1.norm / norm2.norm"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)   # key = .norm  (not .body)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(-1, x.shape[1], h, w)


class GDFN(nn.Module):
    """Gated-Dconv FFN — bias=True, sub-layer named .dwconv"""
    def __init__(self, channels, ffn_expansion_factor=2.66):
        super().__init__()
        hidden = int(channels * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(channels, hidden * 2, 1, bias=True)
        self.dwconv      = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1,   # key = .dwconv
                                     groups=hidden * 2, bias=True)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention — bias=True, dw named .dwconv"""
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv         = nn.Conv2d(channels, channels * 3, 1, bias=True)
        self.dwconv      = nn.Conv2d(channels * 3, channels * 3, 3, 1, 1,  # key = .dwconv
                                     groups=channels * 3, bias=True)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).reshape(b, -1, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = MDTA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn   = GDFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """Single Conv2d downsample — checkpoint key: downN.body"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Single Conv2d upsample — checkpoint key: upN.body"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.ConvTranspose2d(in_ch, out_ch, 2, 2, bias=True)

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    """
    Custom Restormer variant matching checkpoint keys:
      patch_embed  (flat Conv2d, no .proj wrapper)
      encoder1/2/3, down1/2/3
      latent (10 blocks)
      up3/2/1, decoder3/2/1
      output  (with bias)
      NO reduce_chan — skip connections added directly (same channel width after up)
      NO refinement block
    """
    def __init__(self, inp_channels=1, out_channels=1, dim=64):
        super().__init__()
        # patch embed — key: patch_embed.weight / patch_embed.bias
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=True)

        # Encoder
        self.encoder1 = nn.Sequential(*[TransformerBlock(dim,      1) for _ in range(4)])
        self.down1    = Downsample(dim,      dim * 2)

        self.encoder2 = nn.Sequential(*[TransformerBlock(dim * 2,  2) for _ in range(6)])
        self.down2    = Downsample(dim * 2,  dim * 4)

        self.encoder3 = nn.Sequential(*[TransformerBlock(dim * 4,  4) for _ in range(8)])
        self.down3    = Downsample(dim * 4,  dim * 8)

        # Bottleneck
        self.latent   = nn.Sequential(*[TransformerBlock(dim * 8,  8) for _ in range(10)])

        # Decoder — up brings dim*8 → dim*4, then cat with skip → dim*8,
        # but checkpoint has NO reduce_chan, so decoder operates on dim*8 after cat.
        # Sizes verified from checkpoint: decoder3 blocks have dim*8 channels,
        # decoder2 blocks have dim*4, decoder1 blocks have dim*2.
        self.up3      = Upsample(dim * 8,  dim * 4)
        self.decoder3 = nn.Sequential(*[TransformerBlock(dim * 8,  4) for _ in range(8)])

        self.up2      = Upsample(dim * 8,  dim * 2)
        self.decoder2 = nn.Sequential(*[TransformerBlock(dim * 4,  2) for _ in range(6)])

        self.up1      = Upsample(dim * 4,  dim)
        self.decoder1 = nn.Sequential(*[TransformerBlock(dim * 2,  1) for _ in range(4)])

        self.output   = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=True)

    def forward(self, inp):
        x   = self.patch_embed(inp)           # dim

        e1  = self.encoder1(x)                # dim
        e2  = self.encoder2(self.down1(e1))   # dim*2
        e3  = self.encoder3(self.down2(e2))   # dim*4
        lat = self.latent(self.down3(e3))     # dim*8

        # Decoder: upsample then concat skip — no reduce_chan
        d3  = self.decoder3(torch.cat([self.up3(lat), e3], dim=1))   # dim*8
        d2  = self.decoder2(torch.cat([self.up2(d3),  e2], dim=1))   # dim*4
        d1  = self.decoder1(torch.cat([self.up1(d2),  e1], dim=1))   # dim*2

        return self.output(d1) + inp


# ─── Helpers ──────────────────────────────────────────────────────────────────

MODEL_PATH = "Restormer_final.pth"
GDRIVE_ID  = "1LVLEc1e_x5oSBwlaFpSESTKIB0meHDoq"


def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            import gdown
            with st.spinner("⬇️  Downloading model weights from Google Drive…"):
                gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()


@st.cache_resource
def load_model():
    download_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Restormer(inp_channels=1, out_channels=1)
    state  = torch.load(MODEL_PATH, map_location=device)
    # Support plain state_dict or wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, device


def preprocess_mri(image_input, size: int = 256):
    """
    Preprocess a corrupted MRI image for Restormer inference.

    Args:
        image_input: file path (str) OR raw bytes from upload
        size: resize target (default 256)

    Returns:
        tensor: (1, 1, size, size) float32 tensor ready for model
        img_np: (size, size) numpy array for display
    """
    # ---- Load image ----
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            # File path
            img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        else:
            # Raw bytes
            file_bytes = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("input must be file path or bytes")

    if img is None:
        raise ValueError("Failed to load image — check file format")

    # ---- Resize ----
    img = cv2.resize(img, (size, size))

    # ---- Normalize to [0, 1] ----
    img_np = img.astype(np.float32) / 255.0

    # ---- Convert to tensor (1, 1, size, size) ----
    tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    return tensor, img_np


def postprocess_output(output_tensor):
    """
    Convert model output tensor back to displayable numpy image.

    Args:
        output_tensor: (1, 1, H, W) model output

    Returns:
        img_float: numpy array (H, W) float32 in [0, 1]
        img_uint8: numpy array (H, W) uint8 in [0, 255] for saving/display
    """
    img_float = output_tensor.cpu().squeeze().numpy()
    img_float = np.clip(img_float, 0.0, 1.0)
    img_uint8 = (img_float * 255).astype(np.uint8)
    return img_float, img_uint8


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.read()


# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MRI Reconstruction — Restormer",
    page_icon="🧠",
    layout="wide",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background: #050a14;
    color: #e0eaff;
}

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1f3c 0%, #050a14 60%);
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(100,160,255,0.15);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    text-align: center;
}

.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5577aa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #7eb8ff;
}

.badge {
    display: inline-block;
    background: rgba(100,160,255,0.12);
    border: 1px solid rgba(100,160,255,0.3);
    border-radius: 20px;
    padding: 0.15rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #7eb8ff;
    margin: 0.2rem;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: #3a6090;
    margin-bottom: 0.5rem;
}

hr { border-color: rgba(100,160,255,0.1); }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
  <div class="section-label">Neural Image Restoration</div>
  <h1 style="font-size:2.8rem; margin:0; background: linear-gradient(135deg, #7eb8ff 0%, #a78bfa 100%);
     -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    MRI Reconstruction
  </h1>
  <p style="color:#5577aa; font-family:'Space Mono',monospace; font-size:0.8rem; margin-top:0.4rem;">
    Powered by <span style="color:#7eb8ff;">Restormer</span> — Efficient Transformer for High-Resolution Image Restoration
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<span class="badge">Transformer-based</span> <span class="badge">Residual learning</span> <span class="badge">Grayscale MRI</span>', unsafe_allow_html=True)
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    img_size = st.selectbox("Input resolution", [128, 192, 256], index=2)
    st.markdown("---")
    st.markdown("""
    **About Restormer**

    Restormer uses multi-head transposed attention and gated feed-forward networks
    inside a U-Net style encoder-decoder with skip connections — achieving
    state-of-the-art restoration without quadratic complexity.
    """)

# ─── Load model ───────────────────────────────────────────────────────────────
try:
    model, device = load_model()
    st.sidebar.success(f"Model ready ✅  · {str(device).upper()}")
except Exception as e:
    st.sidebar.error(f"Model error: {e}")
    st.stop()

# ─── Upload ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a corrupted MRI image (JPG / PNG / BMP / TIFF)",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded is None:
    st.info("👆 Upload a corrupted MRI image to begin reconstruction.")
    st.stop()

# ─── Preprocess ───────────────────────────────────────────────────────────────
with st.spinner("Preprocessing image…"):
    try:
        corrupted_tensor, corrupted_np = preprocess_mri(uploaded.read(), size=img_size)
    except ValueError as e:
        st.error(f"Could not decode the uploaded image: {e}")
        st.stop()

# ─── Inference ────────────────────────────────────────────────────────────────
with st.spinner("Running Restormer inference…"):
    inp = corrupted_tensor.to(device)
    with torch.no_grad():
        residual = model(inp)
        output   = torch.clamp(inp + residual, 0.0, 1.0)
    recon_np, recon_uint8_inf = postprocess_output(output)

diff_np = np.abs(recon_np - corrupted_np)

# ─── Metrics ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Run Info</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Resolution</div><div class="metric-value">{img_size}²</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Device</div><div class="metric-value">{str(device).upper()}</div></div>', unsafe_allow_html=True)
with c3:
    mean_diff = float(diff_np.mean()) * 100
    st.markdown(f'<div class="metric-card"><div class="metric-label">Mean Δ</div><div class="metric-value">{mean_diff:.2f}%</div></div>', unsafe_allow_html=True)

# ─── Visual comparison ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Visual Comparison</div>', unsafe_allow_html=True)

BG = "#050a14"
fig = plt.figure(figsize=(12, 4.5), facecolor=BG)
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.06, left=0.02, right=0.98)

panels = [
    (corrupted_np, "Corrupted Input",    "gray"),
    (recon_np,     "Reconstructed",      "gray"),
    (diff_np,      "Residual  |Δ|",      "magma"),
]

for i, (img, title, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(BG)
    im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if cmap == "gray" else None, interpolation="bicubic")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#7eb8ff", pad=10, fontfamily="monospace")
    ax.axis("off")
    # Subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3060")
        spine.set_linewidth(1.2)
        spine.set_visible(True)

st.image(fig_to_bytes(fig), use_container_width=True)
plt.close(fig)

# ─── Download ─────────────────────────────────────────────────────────────────
st.markdown("---")
recon_pil   = Image.fromarray(recon_uint8_inf, mode="L")
buf = io.BytesIO()
recon_pil.save(buf, format="PNG")
buf.seek(0)

st.download_button(
    label="⬇️  Download Reconstructed Image",
    data=buf,
    file_name="restormer_reconstructed_mri.png",
    mime="image/png",
)

st.markdown("---")
st.markdown(
    '<p style="font-family:\'Space Mono\',monospace; font-size:0.68rem; color:#2a4060; text-align:center;">'
    "Restormer · Efficient Transformer for High-Resolution Image Restoration · Grayscale MRI mode"
    "</p>",
    unsafe_allow_html=True,
)
