import gradio as gr
import gradio.themes
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from model import Generator as PikaPikaGen
from utils import denormalize_image, plot_attention_visualization
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = "training_output/models/checkpoint_epoch_150.pth"
TOKENIZER_NAME = "prajjwal1/bert-mini"


class PokemonGenerator:
    """Main class for the Pokemon generation demo"""

    def __init__(self):
        self.device = DEVICE
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

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

            # Il modello restituisce tensori con batch_size = num_samples
            generated_256, generated_64, attention_maps, initial_weights = self.generator(
                text_ids, attention_mask, return_attentions=True
            )

            # Converti i tensori in immagini PIL per l'output finale
            output_images = []
            images_to_process = []
            if resolution in ["256x256", "both"]:
                images_to_process.append(generated_256)
            if resolution in ["64x64", "both"]:
                images_to_process.append(generated_64)

            for img_batch in images_to_process:
                # Denormalizza l'intero batch in una sola volta
                img_batch_denorm = denormalize_image(img_batch.cpu())
                for i in range(num_samples):
                    img_pil = self._tensor_to_pil(img_batch_denorm[i])
                    output_images.append(img_pil)

            attention_plot = None
            if show_attention:
                # ### <<< MODIFICA CHIAVE 1: Creare la directory se non esiste
                output_dir = "attention_visualizations"
                os.makedirs(output_dir, exist_ok=True)

                # ### <<< MODIFICA CHIAVE 2: Creare un ID più descrittivo per il file
                # Per evitare di sovrascrivere lo stesso file "demo_0"
                plot_id = description.strip().replace(" ", "_")[:30]


                # Usa il primo campione per la visualizzazione dell'attenzione
                attention_plot = plot_attention_visualization(
                    # Argomenti di identificazione
                    epoch=0, # Epoch non è rilevante in un contesto demo
                    set_name="demo",
                    output_dir=output_dir,

                    # ### <<< MODIFICA CHIAVE 3: Passare il tensore grezzo, non la lista di PIL
                    generated_images=generated_256, # Usiamo l'output ad alta risoluzione

                    # Dati batch completi dal modello
                    decoder_attention_maps=attention_maps,
                    initial_context_weights=initial_weights,

                    # Input batch completo
                    token_ids=text_ids,
                    attention_mask=attention_mask,
                    tokenizer=self.tokenizer,

                    # Metadati per il campione specifico (il primo)
                    description=description,
                    pokemon_id=plot_id,

                    # Opzioni
                    sample_idx=0,
                    show_inline=False
                )
                # La funzione salva un file, ma non restituisce un oggetto plot.
                # Se si volesse mostrare il plot in un'interfaccia (es. Gradio/Streamlit),
                # la funzione plot_attention_visualization dovrebbe restituire il path del file salvato.
                # Per ora, lasciamo attention_plot a None come indicatore.

            return output_images, attention_plot


# Initialize the generator
print("🚀 Initializing PikaPikaGen Demo...")
pokemon_gen = PokemonGenerator()

def generate_pokemon_interface(description, num_samples, show_attention, resolution):
    """Main interface function for Gradio"""
    images, attention_plot = pokemon_gen.generate_pokemon(
        description=description,
        num_samples=num_samples,
        show_attention=show_attention,
        resolution=resolution
    )

    if images is None:
        return [], attention_plot  # attention_plot contains error message in this case

    status_msg = f"✅ Generated {len(images)} Pokemon sprites"
    if resolution == "both":
        status_msg += f" ({num_samples} at 256x256 + {num_samples} at 64x64)"
    else:
        status_msg += f" at {resolution}"

    return images, attention_plot if attention_plot else status_msg

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""

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
            Create unique Pokemon sprites from text descriptions using advanced AI!<br>
            Powered by Transformer attention and CNN generation.
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
                        label="Number of Samples"
                    )

                    resolution = gr.Radio(
                        choices=["256x256", "64x64", "both"],
                        value="256x256",
                        label="Output Resolution"
                    )

                show_attention = gr.Checkbox(
                    label="Show Attention Visualization",
                    value=True,
                    info="Visualize which words the AI focuses on"
                )

                generate_btn = gr.Button(
                    "🎨 Generate Pokemon!",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("### 💡 Tips")
                gr.Markdown("""
                - Be descriptive: mention colors, types, features
                - Try: "electric mouse", "ice dragon", "grass cat"
                - Experiment with combinations!
                - Use Pokemon types: fire, water, grass, electric, etc.
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 🎨 Generated Pokemon")

                output_gallery = gr.Gallery(
                    label="Generated Pokemon Sprites",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto",
                    allow_preview=True
                )

                attention_output = gr.Image(
                    label="Attention Visualization",
                    show_label=True,
                    interactive=False
                )

        # Examples section
        gr.Markdown("### 🌟 Try These Examples")
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
        **PikaPikaGen** - Text-to-Image Pokemon Generation using Transformer + GAN Architecture
        🔬 Research Project | 🎯 Deep Learning | 🎮 AI Art Generation
        """)

    return demo

if __name__ == "__main__":
    print("🎮 Starting PikaPikaGen Demo...")

    if not pokemon_gen.model_loaded:
        print("\n❌ Model not loaded! Please ensure you have:")
        print("1. A trained checkpoint in 'training_output/models/'")
        print("2. The model.py file with PikaPikaGen class")
        print("3. All required dependencies installed")
        print("\nTo train the model, run the PikaPikaGen.ipynb notebook first.")
        exit(1)

    print("✅ Model loaded successfully!")
    print("🚀 Launching Gradio interface...")

    # Create and launch interface
    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=False,
        show_error=True,
        inbrowser=True         # Auto-open browser
    )
