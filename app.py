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

# ─── Restormer Architecture ───────────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x.flatten(2).transpose(1, 2)).transpose(1, 2).view(-1, x.shape[1], h, w)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network"""
    def __init__(self, channels, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(channels * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(channels, hidden * 2, 1, bias=bias)
        self.dw_conv     = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, channels, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dw_conv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention"""
    def __init__(self, channels, num_heads, bias=False):
        super().__init__()
        self.num_heads  = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv        = nn.Conv2d(channels, channels * 3, 1, bias=bias)
        self.qkv_dw     = nn.Conv2d(channels * 3, channels * 3, 3, 1, 1, groups=channels * 3, bias=bias)
        self.project_out = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
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
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn   = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
    ):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])
        self.down1_2         = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])
        self.down2_3         = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])])
        self.down3_4         = Downsample(dim * 4)

        self.latent = nn.Sequential(*[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias) for _ in range(num_blocks[3])])

        self.up4_3          = Upsample(dim * 8)
        self.reduce_chan3   = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])])

        self.up3_2          = Upsample(dim * 4)
        self.reduce_chan2   = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])

        self.up2_1          = Upsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias) for _ in range(num_refinement_blocks)])
        self.output     = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp):
        inp_enc_l1 = self.patch_embed(inp)
        out_enc_l1 = self.encoder_level1(inp_enc_l1)

        inp_enc_l2 = self.down1_2(out_enc_l1)
        out_enc_l2 = self.encoder_level2(inp_enc_l2)

        inp_enc_l3 = self.down2_3(out_enc_l2)
        out_enc_l3 = self.encoder_level3(inp_enc_l3)

        inp_enc_l4 = self.down3_4(out_enc_l3)
        latent     = self.latent(inp_enc_l4)

        inp_dec_l3 = self.up4_3(latent)
        inp_dec_l3 = self.reduce_chan3(torch.cat([inp_dec_l3, out_enc_l3], dim=1))
        out_dec_l3 = self.decoder_level3(inp_dec_l3)

        inp_dec_l2 = self.up3_2(out_dec_l3)
        inp_dec_l2 = self.reduce_chan2(torch.cat([inp_dec_l2, out_enc_l2], dim=1))
        out_dec_l2 = self.decoder_level2(inp_dec_l2)

        inp_dec_l1 = self.up2_1(out_dec_l2)
        inp_dec_l1 = torch.cat([inp_dec_l1, out_enc_l1], dim=1)
        out_dec_l1 = self.decoder_level1(inp_dec_l1)

        out = self.refinement(out_dec_l1)
        return self.output(out) + inp


# ─── Helpers ──────────────────────────────────────────────────────────────────

MODEL_PATH = "Restormer_final.pth"
GDRIVE_ID  = "1VW-F-SLnxhFg1Pvtv5sx44xQYo3c2v5B"


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
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, device


def preprocess(uploaded_file, size: int = 256):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, (size, size))
    img_norm = img.astype(np.float32) / 255.0
    tensor   = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
    return tensor, img_norm


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
    corrupted_tensor, corrupted_np = preprocess(uploaded, size=img_size)

if corrupted_tensor is None:
    st.error("Could not decode the uploaded image. Please try another file.")
    st.stop()

# ─── Inference ────────────────────────────────────────────────────────────────
with st.spinner("Running Restormer inference…"):
    inp = corrupted_tensor.to(device)
    with torch.no_grad():
        out = model(inp)
        recon_dev = torch.clamp(out, 0.0, 1.0)
    recon_np = recon_dev.cpu().squeeze().numpy()

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
recon_uint8 = (recon_np * 255).clip(0, 255).astype(np.uint8)
recon_pil   = Image.fromarray(recon_uint8, mode="L")
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
