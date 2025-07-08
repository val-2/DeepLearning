from training import train

if __name__ == "__main__":
    """
    Punto di ingresso principale per avviare il training del modello.
    """
    print("Avvio dello script principale...")
    train(resume_from_checkpoint=True)
    print("Esecuzione del training completata.")
