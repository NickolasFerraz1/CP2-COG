from flask import Flask, request, jsonify
import joblib
import numpy as np

# Carregar o modelo treinado
modelo = joblib.load("modelo_logistico.pkl")

# Iniciar a aplicação Flask
app = Flask(__name__)

# Rota para verificar o status da API
@app.route("/", methods=["GET"])
def home():
    return "API de Machine Learning no Azure funcionando!"

# Rota para realizar previsões usando o modelo
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obter os dados enviados no corpo da requisição
        dados = request.get_json()
        
        # Converter os dados para um formato adequado
        X = np.array(dados["features"]).reshape(1, -1)
        
        # Realizar a previsão
        previsao = modelo.predict(X)

        # Retornar a previsão como resposta
        return jsonify({"previsao": int(previsao[0])})
    except Exception as e:
        return jsonify({"erro": str(e)})

if __name__ == "__main__":
    # Rodar a aplicação Flask
    app.run(host="0.0.0.0", port=8000)
