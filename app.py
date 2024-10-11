from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado
modelo = joblib.load("modelo_logistico.pkl")

# Rota principal para testar se a API está online
@app.route("/", methods=["GET"])
def home():
    return "API de Machine Learning no Azure funcionando!"

# Rota para previsões
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obter os dados enviados no corpo da requisição
        dados = request.get_json()

        # Verificar se o campo 'features' está presente nos dados
        if "features" not in dados:
            return jsonify({"erro": "Campo 'features' ausente no corpo da requisição"}), 400

        # Convertendo os dados para um numpy array
        X = np.array(dados["features"]).reshape(1, -1)

        # Realizar a previsão
        previsao = modelo.predict(X)

        # Retornar a previsão como resposta JSON
        return jsonify({"previsao": int(previsao[0])})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
