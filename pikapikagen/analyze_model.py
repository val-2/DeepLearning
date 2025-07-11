from model import PikaPikaGen
from discriminator import PikaPikaDisc

def analyze_models(generator, discriminator):
    """
    Stampa un riepilogo dettagliato dei parametri del modello, distinguendo
    tra i vari componenti principali (Generator, Discriminator).
    """
    print("--- Analisi Parametri Modello ---")

    # Funzione helper per contare i parametri di un sottomodulo
    def count_submodule_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    # --- Analisi Generatore ---
    print("\n[--- GENERATOR (PikaPikaGen) ---]")
    g_total, g_trainable = count_submodule_params(generator)

    # Analisi Text Encoder
    encoder_total, encoder_trainable = count_submodule_params(generator.text_encoder)
    print("  [1] Text Encoder:")
    print(f"    - Totali:       {encoder_total:,}")
    print(f"    - Addestrabili: {encoder_trainable:,}")

    # Analisi Image Decoder
    decoder_total, decoder_trainable = count_submodule_params(generator.image_decoder)
    print("  [2] Image Decoder:")
    print(f"    - Totali:       {decoder_total:,}")
    print(f"    - Addestrabili: {decoder_trainable:,}")
    
    print("  ---------------------------------")
    print(f"  Generator Totali:       {g_total:,}")
    print(f"  Generator Addestrabili: {g_trainable:,}")


    # --- Analisi Discriminatore ---
    print("\n[--- DISCRIMINATOR (PikaPikaDisc) ---]")
    d_total, d_trainable = count_submodule_params(discriminator)
    print(f"  - Totali:       {d_total:,}")
    print(f"  - Addestrabili: {d_trainable:,}")


    # --- Riepilogo ---
    total_params = g_total + d_total
    trainable_params = g_trainable + d_trainable

    print("\n" + "="*50)
    print("Riepilogo Complessivo:")
    print(f"  - Parametri Totali:       {total_params:,}")
    print(f"  - Parametri Addestrabili: {trainable_params:,}")
    print("="*50)


def main():
    """
    Funzione principale per istanziare i modelli e lanciare l'analisi.
    """
    # Impostazioni coerenti con training.py
    model_name = "prajjwal1/bert-mini"
    noise_dim = 100
    text_embed_dim = 256
    fine_tune_embeddings = True # Come da default nel training

    # Istanzia il generatore
    model_G = PikaPikaGen(
        text_encoder_model_name=model_name,
        noise_dim=noise_dim,
        fine_tune_embeddings=fine_tune_embeddings
    )

    # Istanzia il discriminatore
    model_D = PikaPikaDisc(text_embed_dim=text_embed_dim)

    analyze_models(model_G, model_D)

if __name__ == "__main__":
    main()
