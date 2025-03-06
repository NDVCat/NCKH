import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Tải mô hình đã huấn luyện
try:
    model = joblib.load('oilrate_model.pkl')
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            data = request.get_json(force=True)
        else:  # Nếu là GET, lấy dữ liệu từ query parameters
            data = request.args.to_dict()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Chuyển đổi giá trị từ string -> float
        data = {key: float(value) for key, value in data.items()}

        print("Dữ liệu nhận được:", data)  # Debugging

        # Chuyển đổi dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        return jsonify({'Predicted_Oilrate': prediction[0]})
    except Exception as e:
        print("Lỗi:", str(e))  # Debugging
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
