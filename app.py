from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado (o arquivo modelo_logistico.pkl deve estar no mesmo diretório)
modelo = joblib.load("modelo_logistico.pkl")

@app.route("/test_post", methods=["POST"])
def test_post():
    return jsonify({"message": "POST funcionando!"}), 200

# Rota principal para previsão
@app.route("/", methods=["POST"])
def predict():
    try:
        # Obter os dados enviados no corpo da requisição
        dados = request.get_json()

        # Verificar se o campo 'features' está presente nos dados
        if "features" not in dados:
            return jsonify({"erro": "Campo 'features' ausente no corpo da requisição"}), 400

        # Convertendo os dados para numpy array
        X = np.array(dados["features"]).reshape(1, -1)

        # Realizar a previsão
        previsao = modelo.predict(X)

        # Retornar a previsão como resposta JSON
        return jsonify({"previsao": int(previsao[0])})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# Executar o aplicativo Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
