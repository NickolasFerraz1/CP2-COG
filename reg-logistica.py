import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Geração de um conjunto de dados exemplo (você pode usar seu próprio dataset)
def create_and_train_model():
    # Conjunto de dados sintético (Iris dataset)
    from sklearn.datasets import load_iris
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo de Regressão Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Salvar o modelo treinado em um arquivo para futura utilização
    joblib.dump(model, "modelo_logistico.pkl")
    print("Modelo treinado e salvo com sucesso!")

if __name__ == "__main__":
    create_and_train_model()
