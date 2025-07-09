from training import train

if __name__ == "__main__":
    """
    Punto di ingresso principale per avviare il training del modello.
    """
    print("Avvio dello script principale...")
    train(continue_from_last_checkpoint=True, epochs_to_run=100)
    print("Esecuzione del training completata.")
