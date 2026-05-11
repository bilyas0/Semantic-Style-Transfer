from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.semantic_style_transfer import SemanticStyleTransferPipeline


# ============================================================
# OTOMATİK DEFAULT AYARLAR
# Colab'da iyi çalışan değerler burada sabit tutulur.
# Kullanıcı bunları görmez.
# ============================================================

IMAGE_SIZE = 384
NST_STEPS = 150
CONTENT_WEIGHT = 5.0
STYLE_WEIGHT = 100000.0
MASK_BLUR = 21
LEARNING_RATE = 0.03


# ============================================================
# KLASÖR AYARLARI
# Colab'daki inputs / outputs mantığının Streamlit hali
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def prepare_uploaded_image(uploaded_file, save_path: Path, image_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Streamlit'ten yüklenen görseli okur, RGB'ye çevirir,
    sabit boyuta getirir ve PNG olarak kaydeder.

    Colab'daki:
        Image.open(path).convert("RGB")
        img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img.save(...)
    mantığının otomatikleştirilmiş halidir.
    """

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((image_size, image_size))

    image.save(save_path)

    return image


def pil_to_png_bytes(image: Image.Image) -> bytes:
    """
    PIL görselini indirilebilir PNG byte verisine çevirir.
    """

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ============================================================
# STREAMLIT SAYFA AYARI
# ============================================================

st.set_page_config(
    page_title="Semantic Style Transfer",
    layout="wide"
)

st.title("Semantic Style Transfer")
st.write(
    "Content fotoğrafını ve style görselini yükle. "
    "Uygulama görselleri otomatik hazırlar ve semantic style transfer sonucunu üretir."
)


# ============================================================
# DOSYA YÜKLEME ALANI
# ============================================================

st.markdown("## Görselleri Yükle")

col1, col2 = st.columns(2)

with col1:
    content_file = st.file_uploader(
        "Content fotoğrafı",
        type=["jpg", "jpeg", "png"],
        help="İnsan olan ana fotoğrafı yükle."
    )

with col2:
    style_file = st.file_uploader(
        "Style görseli",
        type=["jpg", "jpeg", "png"],
        help="Stil alınacak tablo/desen/sanat görselini yükle."
    )


with st.expander("Kullanım bilgisi"):
    st.markdown(
        """
        **Desteklenen formatlar:** `.jpg`, `.jpeg`, `.png`

        **Content fotoğrafı:** İnsan olan ana fotoğraf olmalı.  
        **Style görseli:** Van Gogh, Monet, Picasso, tablo, desen veya renkli sanat görseli olabilir.

        Görseller otomatik olarak:
        - RGB formatına çevrilir
        - `384 x 384` boyutuna getirilir
        - `inputs/content.png` ve `inputs/style.png` olarak kaydedilir
        - Default parametrelerle çalıştırılır

        Kullanıcının ekstra ayar yapmasına gerek yoktur.
        """
    )


# ============================================================
# GÖRSELLERİ HAZIRLA VE ÖNİZLE
# ============================================================

content_image = None
style_image = None

if content_file is not None:
    content_image = prepare_uploaded_image(
        uploaded_file=content_file,
        save_path=INPUT_DIR / "content.png",
        image_size=IMAGE_SIZE
    )

if style_file is not None:
    style_image = prepare_uploaded_image(
        uploaded_file=style_file,
        save_path=INPUT_DIR / "style.png",
        image_size=IMAGE_SIZE
    )


if content_image is not None or style_image is not None:
    st.markdown("## Yüklenen Görseller")

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        if content_image is not None:
            st.image(
                content_image,
                caption="Content Image",
                use_container_width=True
            )

    with pcol2:
        if style_image is not None:
            st.image(
                style_image,
                caption="Style Image",
                use_container_width=True
            )


# ============================================================
# ÇALIŞTIRMA BUTONU
# ============================================================

run_button = st.button(
    "Style Transfer Başlat",
    type="primary",
    use_container_width=True
)


# ============================================================
# MODELİ ÇALIŞTIR
# ============================================================

if run_button:
    if content_image is None or style_image is None:
        st.warning("Önce content fotoğrafını ve style görselini yüklemelisin.")
        st.stop()

    with st.spinner("Model çalışıyor. Bu işlem bilgisayar gücüne göre biraz sürebilir..."):
        pipeline = SemanticStyleTransferPipeline(
            image_size=IMAGE_SIZE
        )

        result = pipeline.run(
            content_image=content_image,
            style_image=style_image,
            steps=NST_STEPS,
            content_weight=CONTENT_WEIGHT,
            style_weight=STYLE_WEIGHT,
            blur_size=MASK_BLUR,
            learning_rate=LEARNING_RATE,
        )

    st.success("Style Transfer tamamlandı.")

    # ========================================================
    # ÇIKTILARI KAYDET
    # ========================================================

    person_mask_path = OUTPUT_DIR / "person_mask.png"
    full_nst_path = OUTPUT_DIR / "full_style_transfer.png"
    semantic_result_path = OUTPUT_DIR / "semantic_style_transfer_result.png"

    result["person_mask"].save(person_mask_path)
    result["full_styled_image"].save(full_nst_path)
    result["semantic_image"].save(semantic_result_path)

    # ========================================================
    # SONUÇLARI GÖSTER
    # ========================================================

    st.markdown("## Sonuçlar")

    rcol1, rcol2, rcol3 = st.columns(3)

    with rcol1:
        st.image(
            result["person_mask"],
            caption="Person Mask",
            use_container_width=True
        )

    with rcol2:
        st.image(
            result["full_styled_image"],
            caption="Full Neural Style Transfer",
            use_container_width=True
        )

    with rcol3:
        st.image(
            result["semantic_image"],
            caption="Semantic Style Transfer",
            use_container_width=True
        )

    st.markdown("## Final Sonuç")

    st.image(
        result["semantic_image"],
        caption="Final Semantic Style Transfer Output",
        use_container_width=True
    )

    st.download_button(
        label="Final sonucu indir",
        data=pil_to_png_bytes(result["semantic_image"]),
        file_name="semantic_style_transfer_result.png",
        mime="image/png",
        use_container_width=True
    )

    with st.expander("Kaydedilen dosyalar"):
        st.code(
            f"""
inputs/content.png
inputs/style.png

outputs/person_mask.png
outputs/full_style_transfer.png
outputs/semantic_style_transfer_result.png
""",
            language="text"
        )

    with st.expander("Teknik metrikler"):
        st.json(result.get("metrics", {}))