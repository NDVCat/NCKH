import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os

MODEL_PATH = "oilrate_model.pkl"

# Tải mô hình nếu tồn tại
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found!")
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

model = load_model()

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model is not available'}), 500

        data = request.get_json(silent=True) if request.method == 'POST' else request.args.to_dict()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Kiểm tra và chuyển đổi dữ liệu
        try:
            data = {key: float(value) for key, value in data.items()}
        except ValueError:
            return jsonify({'error': 'Invalid input data, must be numbers'}), 400

        # Chuyển thành DataFrame và dự đoán
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)

        return jsonify({'Predicted_Oilrate': float(prediction[0])})  # Định dạng kết quả hợp lệ
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Xử lý lỗi chung
@app.errorhandler(500)
def handle_500_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Mặc định là 5000
    print(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)