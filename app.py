"""
app.py - Cozy Flooring AI Visualiser

Self-hosted AI flooring visualiser for Cozy Flooring Co.
- Upload a room photo
- Choose one of your real Cozy products
- Generate AI floor preview
- View product / order sample
"""

from io import BytesIO
from typing import Optional, Tuple
from urllib.request import Request, urlopen

import gradio as gr
import numpy as np
from PIL import Image

from core import load_model, process_room

model = None
processor = None
device = None

PRODUCTS = {
    "harbor-twist-pebble": {
        "name": 'Harbor Twist – Pebble',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22139045.jpg',
        "price": '£12.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-pebble',
        "sample_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-pebble-sample',
    },
    "harbor-twist-granite": {
        "name": 'Harbor Twist – Granite',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22139088.jpg',
        "price": '£12.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-granite',
        "sample_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-granite-sample',
    },
    "harbor-twist-charcoal": {
        "name": 'Harbor Twist – Charcoal',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22139100.jpg',
        "price": '£12.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-charcoal',
        "sample_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-charcoal-sample',
    },
    "harbor-twist-sand": {
        "name": 'Harbor Twist – Sand',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138944.jpg',
        "price": '£12.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-sand',
        "sample_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-sand-sample',
    },
    "harbor-twist-taupe": {
        "name": 'Harbor Twist – Taupe',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138961.jpg',
        "price": '£12.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-taupe',
        "sample_url": 'https://cozyflooringstore.co.uk/products/harbor-twist-taupe-sample',
    },
    "homestead-festival": {
        "name": 'Homestead – Festival',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22140477.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/homestead-festival',
        "sample_url": 'https://cozyflooringstore.co.uk/products/homestead-festival-sample',
    },
    "homestead-solstice": {
        "name": 'Homestead – Solstice',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22140612.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/homestead-solstice',
        "sample_url": 'https://cozyflooringstore.co.uk/products/homestead-solstice-sample',
    },
    "homestead-celtic": {
        "name": 'Homestead – Celtic',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22140574.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/homestead-celtic',
        "sample_url": 'https://cozyflooringstore.co.uk/products/homestead-celtic-sample',
    },
    "homestead-oak": {
        "name": 'Homestead – Oak',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22140493.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/homestead-oak',
        "sample_url": 'https://cozyflooringstore.co.uk/products/homestead-oak-sample',
    },
    "aura-desert": {
        "name": 'Aura – Desert',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22141147.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/aura-desert',
        "sample_url": 'https://cozyflooringstore.co.uk/products/aura-desert-sample',
    },
    "aura-cloud": {
        "name": 'Aura – cloud',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22141244.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/aura-cloud',
        "sample_url": 'https://cozyflooringstore.co.uk/products/aura-cloud-sample',
    },
    "aura-silver": {
        "name": 'Aura – silver',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22141261.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/aura-silver',
        "sample_url": 'https://cozyflooringstore.co.uk/products/aura-silver-sample',
    },
    "aura-fog": {
        "name": 'Aura – fog',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22141201.jpg',
        "price": '£15.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/aura-fog',
        "sample_url": 'https://cozyflooringstore.co.uk/products/aura-fog-sample',
    },
    "sandhaven-steel": {
        "name": 'Sandhaven – Steel',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC21612693.jpg',
        "price": '£19.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/sandhaven-steel',
        "sample_url": 'https://cozyflooringstore.co.uk/products/sandhaven-steel-sample',
    },
    "sandhaven-beige": {
        "name": 'Sandhaven – beige',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC21612472.jpg',
        "price": '£19.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/sandhaven-beige',
        "sample_url": 'https://cozyflooringstore.co.uk/products/sandhaven-beige-sample',
    },
    "sandhaven-grey": {
        "name": 'Sandhaven – grey',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC21612596.jpg',
        "price": '£19.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/sandhaven-grey',
        "sample_url": 'https://cozyflooringstore.co.uk/products/sandhaven-grey-sample',
    },
    "sandhaven-blue": {
        "name": 'Sandhaven – blue',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC21612511.jpg',
        "price": '£19.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/sandhaven-blue',
        "sample_url": 'https://cozyflooringstore.co.uk/products/sandhaven-blue-sample',
    },
    "sandhaven-silver": {
        "name": 'Sandhaven – silver',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC21612677.jpg',
        "price": '£19.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/sandhaven-silver',
        "sample_url": 'https://cozyflooringstore.co.uk/products/sandhaven-silver-sample',
    },
    "serenity-carbon": {
        "name": 'Serenity – Carbon',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138022.jpg',
        "price": '£22.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/serenity-carbon',
        "sample_url": 'https://cozyflooringstore.co.uk/products/serenity-carbon-sample',
    },
    "serenity-latte": {
        "name": 'Serenity – latte',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138081.jpg',
        "price": '£22.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/serenity-latte',
        "sample_url": 'https://cozyflooringstore.co.uk/products/serenity-latte-sample',
    },
    "serenity-cashmere": {
        "name": 'Serenity – cashmere',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138049.jpg',
        "price": '£22.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/serenity-cashmere',
        "sample_url": 'https://cozyflooringstore.co.uk/products/serenity-cashmere-sample',
    },
    "serenity-sage": {
        "name": 'Serenity – sage',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138235.jpg',
        "price": '£22.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/serenity-sage',
        "sample_url": 'https://cozyflooringstore.co.uk/products/serenity-sage-sample',
    },
    "serenity-seal": {
        "name": 'Serenity – seal',
        "texture": 'https://www.hfdscotland.co.uk/prodimages/detailzoom/GPC22138189.jpg',
        "price": '£22.99/m²',
        "product_url": 'https://cozyflooringstore.co.uk/products/serenity-seal',
        "sample_url": 'https://cozyflooringstore.co.uk/products/serenity-seal-sample',
    },
    "cozy-product-url": {
        "name": 'Cozy product URL:',
        "texture": 'Price per m²:',
        "price": '£Sample URL:/m²',
        "product_url": 'Texture image URL:',
        "sample_url": 'Visualiser handle:',
    },
}


DEFAULT_PRODUCT_HANDLE = next(iter(PRODUCTS.keys()))


def product_label(handle: str) -> str:
    product = PRODUCTS[handle]
    return f"{product['name']} — {product['price']}"


PRODUCT_LABELS = {product_label(handle): handle for handle in PRODUCTS}
PRODUCT_CHOICES = list(PRODUCT_LABELS.keys())
DEFAULT_PRODUCT_LABEL = product_label(DEFAULT_PRODUCT_HANDLE)


def resolve_product_handle(product_choice: str) -> str:
    if product_choice in PRODUCTS:
        return product_choice
    return PRODUCT_LABELS.get(product_choice, DEFAULT_PRODUCT_HANDLE)


def clean_url(url: str) -> str:
    return (url or "").strip()


def initialize_model():
    global model, processor, device
    if model is None:
        model, processor, device = load_model()
    return model, processor, device


def load_image_from_url(url: str) -> Optional[np.ndarray]:
    """Download image from URL using stdlib only."""
    url = clean_url(url)
    if not url:
        return None
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=25) as response:
            data = response.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise gr.Error(f"Could not load texture image. Check the image link. Details: {exc}")


def ensure_rgb(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    if image.ndim == 2:
        image = np.dstack([image, image, image])
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    return image.astype(np.uint8)


def product_markdown(product_choice: str) -> str:
    handle = resolve_product_handle(product_choice)
    p = PRODUCTS[handle]
    return f"""
### {p['name']}
**{p['price']}**

[View product]({p['product_url']}) &nbsp; | &nbsp; [Order sample]({p['sample_url']})

Visualiser handle: `{handle}`
"""


def get_initial_product(request: gr.Request):
    """Auto-select product from /?product=handle when embedded from Shopify."""
    handle = DEFAULT_PRODUCT_HANDLE
    try:
        query_handle = request.query_params.get("product") if request else None
        if query_handle and query_handle in PRODUCTS:
            handle = query_handle
    except Exception:
        pass
    label = product_label(handle)
    return label, product_markdown(label)


def visualize_floor(
    room_image: np.ndarray,
    product_choice: str,
    custom_texture_image: np.ndarray,
    custom_texture_url: str,
    blend_strength: float = 0.62,
    tile_size: int = 90,
    edge_feather: int = 17,
    mask_dilate: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    global model, processor, device
    if model is None:
        initialize_model()

    room_image = ensure_rgb(room_image)
    if room_image is None:
        raise gr.Error("Upload a room photo first.")

    handle = resolve_product_handle(product_choice)
    selected_product = PRODUCTS[handle]

    # Priority order:
    # 1. Custom pasted URL, 2. Custom uploaded texture, 3. Selected Cozy product texture.
    custom_texture_url = clean_url(custom_texture_url)
    if custom_texture_url:
        texture_image = load_image_from_url(custom_texture_url)
    elif custom_texture_image is not None:
        texture_image = ensure_rgb(custom_texture_image)
    else:
        texture_image = load_image_from_url(selected_product["texture"])

    if texture_image is None:
        raise gr.Error("Choose a Cozy product, upload a texture, or paste a texture URL.")

    result, mask = process_room(
        room_image,
        texture_image,
        model,
        processor,
        device,
        blend_strength=blend_strength,
        tile_size=int(tile_size),
        edge_feather=int(edge_feather),
        mask_dilate=int(mask_dilate),
    )

    mask_vis = np.zeros_like(room_image)
    mask_vis[:, :, 1] = mask * 255
    alpha = 0.45
    mask_overlay = (room_image * (1 - alpha) + mask_vis * alpha).astype(np.uint8)

    return result, mask_overlay


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

body, .gradio-container { font-family: 'Outfit', sans-serif !important; }
.gradio-container { background: #f5f1eb !important; color: #24211d !important; }
.cozy-header { text-align: center; padding: 28px 0 18px; }
.cozy-kicker { display: inline-block; background: #fff; border: 1px solid #e3d4c1; color: #8a5c2f; padding: 7px 14px; border-radius: 999px; font-size: 13px; font-weight: 700; margin-bottom: 14px; }
.cozy-title { font-size: clamp(34px, 5vw, 58px); line-height: 1; font-weight: 800; color: #191714; margin-bottom: 12px; }
.cozy-subtitle { max-width: 760px; margin: 0 auto; color: #6e6258; font-size: 18px; }
.group-container { background: #fff !important; border: 1px solid #eadfce !important; border-radius: 22px !important; padding: 20px !important; box-shadow: 0 18px 42px rgba(42, 32, 20, 0.08) !important; }
#viz-btn { background: linear-gradient(135deg, #b38b59 0%, #8a5c2f 100%) !important; color: white !important; border: none !important; font-size: 18px !important; font-weight: 800 !important; border-radius: 14px !important; padding: 16px !important; }
#viz-btn:hover { filter: brightness(1.03); transform: translateY(-1px); }
.block-label, label, .wrap span { color: #5f5660 !important; }
.gradio-image, .image-container, .input-image, .output-image { border-radius: 16px !important; }
.product-box { background:#fff8ed; border:1px solid #eadfce; border-radius:16px; padding:14px 16px; }
.footer-note { text-align:center; color:#7b6e62; padding: 24px 0; font-size: 14px; }
"""


def create_interface():
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="stone",
        neutral_hue="stone",
        font=["Outfit", "sans-serif"],
    )

    with gr.Blocks(title="Cozy Flooring Visualiser", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="cozy-header">
              <div class="cozy-kicker">Cozy Flooring Co • AI Room Preview</div>
              <div class="cozy-title">See it in your room</div>
              <div class="cozy-subtitle">
                Upload a room photo, choose a real Cozy product, and preview how the floor could look before ordering samples.
              </div>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=4, variant="panel", elem_classes=["group-container"]):
                gr.Markdown("### 1. Upload room + choose flooring")
                room_input = gr.Image(label="Room photo", type="numpy", height=330)

                product_dropdown = gr.Dropdown(
                    choices=PRODUCT_CHOICES,
                    value=DEFAULT_PRODUCT_LABEL,
                    label="Choose Cozy product",
                    interactive=True,
                )
                product_info = gr.Markdown(product_markdown(DEFAULT_PRODUCT_LABEL), elem_classes=["product-box"])

                with gr.Accordion("Optional: test a custom texture instead", open=False):
                    custom_texture_input = gr.Image(label="Upload custom flooring texture", type="numpy", height=220)
                    custom_texture_url = gr.Textbox(
                        label="Or paste custom texture image URL",
                        placeholder="https://www.hfdscotland.co.uk/prodimages/detailzoom/example.jpg",
                    )

                with gr.Accordion("Fine-tune result", open=True):
                    blend_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.62,
                        step=0.05,
                        label="Keep original shadows",
                        info="Higher = more room lighting/shadows preserved. Lower = stronger new texture.",
                    )
                    tile_slider = gr.Slider(
                        minimum=35,
                        maximum=220,
                        value=90,
                        step=5,
                        label="Texture scale",
                        info="Lower = smaller/tighter carpet fibres. Higher = larger pattern.",
                    )
                    feather_slider = gr.Slider(
                        minimum=3,
                        maximum=41,
                        value=17,
                        step=2,
                        label="Edge feather",
                        info="Softens edges around furniture/skirting so it looks less pasted.",
                    )
                    dilate_slider = gr.Slider(
                        minimum=0,
                        maximum=8,
                        value=2,
                        step=1,
                        label="Mask expansion",
                        info="Slightly expands the detected floor if edges are missed.",
                    )

                process_btn = gr.Button("✨ Generate Cozy Preview", variant="primary", elem_id="viz-btn")

            with gr.Column(scale=5, variant="panel", elem_classes=["group-container"]):
                gr.Markdown("### 2. Preview result")
                result_output = gr.Image(label="Final preview", type="numpy", height=540, interactive=False)
                with gr.Accordion("Check AI floor mask", open=False):
                    mask_output = gr.Image(label="Detected floor area", type="numpy", height=280, interactive=False)

        gr.HTML(
            """
            <div class="footer-note">
              Built for Cozy Flooring Co. Self-hosted AI visualiser test — no monthly visualiser subscription needed.
            </div>
            """
        )

        product_dropdown.change(
            fn=product_markdown,
            inputs=product_dropdown,
            outputs=product_info,
        )

        demo.load(
            fn=get_initial_product,
            inputs=None,
            outputs=[product_dropdown, product_info],
        )

        process_btn.click(
            fn=visualize_floor,
            inputs=[
                room_input,
                product_dropdown,
                custom_texture_input,
                custom_texture_url,
                blend_slider,
                tile_slider,
                feather_slider,
                dilate_slider,
            ],
            outputs=[result_output, mask_output],
        )

    return demo


def main():
    print("=" * 50)
    print("Cozy Flooring Visualiser - Starting")
    print("=" * 50)
    print(f"Loaded {len(PRODUCTS)} Cozy products")

    try:
        initialize_model()
        print("Model loaded successfully!")
    except Exception as exc:
        print(f"Warning: {exc}")
        print("Model will be loaded on first use.")

    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)


if __name__ == "__main__":
    main()
