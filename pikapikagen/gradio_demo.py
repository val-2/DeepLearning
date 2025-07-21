import gradio as gr
import gradio.themes
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from model import Generator as PikaPikaGen
from utils import denormalize_image
from plots import plot_attention_visualization
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "model_checkpoints/checkpoint_epoch_150.pth"
TOKENIZER_NAME = "prajjwal1/bert-mini"


class PokemonGenerator:
    """Main class for the Pokemon generation demo"""

    def __init__(self):
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        self._load_model()


    def _load_model(self):
        """Load the trained PikaPikaGen model"""
        try:
            # Initialize model
            self.generator = PikaPikaGen().to(self.device)

            # Load checkpoint
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)

            # Load saved weights into model
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"✅ Generator loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")

            # No training
            self.generator.eval()

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        # tensor shape: (3, H, W)
        img_np = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def generate_pokemon(self, description, num_samples=4, show_attention=False, resolution="both"):
        """
        Generate Pokemon sprites from text description

        Args:
            description (str): Text description of the desired Pokemon
            num_samples (int): Number of samples to generate (1-8)
            show_attention (bool): Whether to show attention visualization
            resolution (str): Output resolution - "256x256", "64x64", or "both"

        Returns:
            tuple: (generated_images, attention_plot)
        """
        if not description.strip():
            return [], "❌ Please enter a description."

        # No reason to compute gradients
        with torch.no_grad():
            tokens = self.tokenizer(
                description,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            text_ids = tokens['input_ids'].repeat(num_samples, 1).to(self.device)
            attention_mask = tokens['attention_mask'].repeat(num_samples, 1).to(self.device)

            generated_256, generated_64, attention_maps, initial_weights = self.generator(
                text_ids, attention_mask, return_attentions=True
            )

            # Convert tensors to PIL images
            output_images = []
            images_to_process = []
            if resolution in ["256x256", "both"]:
                images_to_process.append(generated_256)
            if resolution in ["64x64", "both"]:
                images_to_process.append(generated_64)

            for img_batch in images_to_process:
                img_batch_denorm = denormalize_image(img_batch.cpu())
                for i in range(num_samples):
                    img_pil = self._tensor_to_pil(img_batch_denorm[i])
                    output_images.append(img_pil)

            attention_plot = None
            if show_attention:
                # Create directory if it doesn't exist
                output_dir = "attention_visualizations"
                os.makedirs(output_dir, exist_ok=True)

                # Create a more descriptive ID for the file
                # To avoid overwriting the same file with the same name
                plot_id = description.strip().replace(" ", "_")[:30]


                # Use the first sample for the attention visualization
                attention_plot = plot_attention_visualization(
                    epoch=0,
                    set_name="demo",
                    output_dir=output_dir,

                    generated_images=generated_256,

                    # Full batch data from the model
                    decoder_attention_maps=attention_maps,
                    initial_context_weights=initial_weights,

                    token_ids=text_ids,
                    attention_mask=attention_mask,
                    tokenizer=self.tokenizer,

                    # Metadata for the specific sample
                    description=description,
                    pokemon_id=plot_id,

                    sample_idx=0,
                    show_inline=False
                )

            return output_images, attention_plot


print("Initializing PikaPikaGen Demo...")
pokemon_gen = PokemonGenerator()

def generate_pokemon_interface(description, num_samples, show_attention, resolution):
    images, attention_plot = pokemon_gen.generate_pokemon(
        description=description,
        num_samples=num_samples,
        show_attention=show_attention,
        resolution=resolution
    )

    if images is None:
        return [], attention_plot  # attention_plot contains error message if error

    status_msg = f"Generated {len(images)} Pokemon sprites"
    if resolution == "both":
        status_msg += f" ({num_samples} at 256x256 + {num_samples} at 64x64)"
    else:
        status_msg += f" at {resolution}"

    return images, attention_plot

def create_interface():
    with gr.Blocks(
        title="PikaPikaGen: AI Pokemon Generator",
        theme=gradio.themes.Soft(),
        css="""
        .main-header {
            text-align: center;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .description {
            text-align: center;
            font-size: 1.1em;
            color: #666;
            margin-bottom: 1em;
        }
        """
    ) as demo:

        gr.HTML("""
        <div class="main-header">🎮 PikaPikaGen: AI Pokemon Generator</div>
        <div class="description">
            Creation of Pokemon sprites from text descriptions using Transformer attention and CNN generation.
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Input")

                description_input = gr.Textbox(
                    label="Pokemon Description",
                    placeholder="Describe your Pokemon! e.g., 'A fire dragon with golden scales and ruby eyes'",
                    lines=3,
                    value="A legendary fire dragon pokemon with golden scales and red eyes"
                )

                with gr.Row():
                    num_samples = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1,
                        label="Number of samples"
                    )

                    resolution = gr.Radio(
                        choices=["256x256", "64x64", "both"],
                        value="256x256",
                        label="Output resolution"
                    )

                show_attention = gr.Checkbox(
                    label="Show attention visualization",
                    value=True,
                    info="Visualize which words the model focuses on"
                )

                generate_btn = gr.Button(
                    "🎨 Generate Pokemon!",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🎨 Generated Pokemon")

                output_gallery = gr.Gallery(
                    label="Generated Pokemon sprites",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    allow_preview=True
                )

                attention_output = gr.Image(
                    label="Attention visualization",
                    show_label=True,
                    interactive=False
                )

        # Examples section
        gr.Markdown("### 🌟 Examples to try")
        gr.Examples(
            examples=[
                ["A fire dragon with golden scales and red eyes", 4, True, "256x256"],
                ["An electric mouse with yellow fur and lightning bolts", 3, False, "both"],
                ["A water turtle with blue shell and powerful jaws", 2, True, "256x256"],
                ["A psychic cat with purple fur and mystical powers", 4, True, "256x256"],
                ["A grass serpent with emerald scales and vine whips", 3, False, "64x64"],
                ["An ice phoenix with crystal wings and frozen flames", 4, True, "256x256"],
                ["A dark wolf with shadow abilities and glowing eyes", 2, True, "both"],
                ["A steel robot pokemon with metallic armor and laser beams", 3, False, "256x256"]
            ],
            inputs=[description_input, num_samples, show_attention, resolution],
            outputs=[output_gallery, attention_output],
            fn=generate_pokemon_interface,
            cache_examples=False
        )

        # Event handlers
        generate_btn.click(
            fn=generate_pokemon_interface,
            inputs=[description_input, num_samples, show_attention, resolution],
            outputs=[output_gallery, attention_output]
        )

        # Footer
        gr.Markdown("""
        ---
        **PikaPikaGen** - Text-to-Image Pokemon Generation using Transformer + CNN
        """)

    return demo

if __name__ == "__main__":
    print("Starting PikaPikaGen Demo...")

    # Create and launch interface
    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        share=False,            # Set to True for public sharing
        debug=False,
        show_error=True,
        inbrowser=True         # Auto-open browser
    )
