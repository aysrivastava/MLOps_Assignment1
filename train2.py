from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    model = KernelRidge(alpha=1.0, kernel='linear')
    trained_model = train_model(model, X_train, y_train)
    
    mse, predictions = evaluate_model(trained_model, X_test, y_test)
    
    print(f"Kernel Ridge MSE: {mse:.4f}")

if __name__ == "__main__":
    main()