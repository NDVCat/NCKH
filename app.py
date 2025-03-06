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

@app.route('/predict', methods=['GET'])
def predict():
    try:
        data = request.get_json(force=True)

        # Kiểm tra dữ liệu đầu vào có đúng định dạng không
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid input format, expected a JSON object'}), 400
        
        # Chuyển đổi dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Trả về kết quả dự đoán
        return jsonify({'Predicted_Oilrate': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
        
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
