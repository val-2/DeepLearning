from model import PikaPikaGen

def analyze_model_parameters(model):
    """
    Stampa un riepilogo dettagliato dei parametri del modello, distinguendo
    tra i vari componenti principali (Text Encoder, Image Decoder).
    """
    print("--- Analisi Parametri Modello PikaPikaGen ---")

    # Parametri totali
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Funzione helper per contare i parametri di un sottomodulo
    def count_submodule_params(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    # Analisi Text Encoder
    encoder_total, encoder_trainable = count_submodule_params(model.text_encoder)
    print("\n[1] Text Encoder:")
    print(f"  - Totali:       {encoder_total:,}")
    print(f"  - Addestrabili: {encoder_trainable:,}")

    # Analisi Image Decoder
    decoder_total, decoder_trainable = count_submodule_params(model.image_decoder)
    print("\n[2] Image Decoder:")
    print(f"  - Totali:       {decoder_total:,}")
    print(f"  - Addestrabili: {decoder_trainable:,}")


    print("\n" + "="*50)
    print("Riepilogo:")
    print(f"  - Parametri Totali:       {total_params:,}")
    print(f"  - Parametri Addestrabili: {trainable_params:,}")
    print("="*50)


def main():
    """
    Funzione principale per istanziare il modello e lanciare l'analisi.
    """
    # Impostazioni coerenti con training.py
    model_name = "prajjwal1/bert-mini"
    noise_dim = 100
    fine_tune_embeddings = True # Come da default nel training

    # Istanzia il modello
    model = PikaPikaGen(
        text_encoder_model_name=model_name,
        noise_dim=noise_dim,
        fine_tune_embeddings=fine_tune_embeddings
    )

    analyze_model_parameters(model)

if __name__ == "__main__":
    main()
