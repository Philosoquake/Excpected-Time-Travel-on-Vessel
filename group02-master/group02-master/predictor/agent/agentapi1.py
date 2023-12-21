import pandas as pd
from predictor.agent.agentt import VesselTravelTimePredictor

def main():
    predictor = VesselTravelTimePredictor()

    while True:
        print("Choose an operation:")
        print("1: Load and preprocess data")
        print("2: Train model")
        print("3: Predict")
        print("4: Save model")
        print("5: Load model")
        print("6: Exit")

        choice = input("> ")

        if choice == '1':
            filepath = input("Enter file path: ")
            predictor.load_data(filepath)
            predictor.preprocess()
        elif choice == '2':
            predictor.train()
        elif choice == '3':
            new_data = input("Enter new data as a dictionary: ")
            new_data = pd.DataFrame(eval(new_data))  # Convert string to dictionary then to DataFrame
            predictor.predict(new_data)
        elif choice == '4':
            path = input("Enter save path: ")
            predictor.save_model(path)
        elif choice == '5':
            path = input("Enter model path: ")
            predictor.load_model(path)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
