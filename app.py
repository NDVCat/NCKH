import joblib
from flask import Flask, request, jsonify, send_file
import pandas as pd
import os

# Tải mô hình
if not os.path.exists('oilrate_model.pkl'):
    print("Error: Model file not found!")
    exit(1)

try:
    model = joblib.load('oilrate_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    exit(1)

app = Flask(__name__)

@app.route('/predict_excel', methods=['POST'])
def predict_excel():
    try:
        data = request.get_json(force=True)

        if not isinstance(data, list):
            return jsonify({'error': 'Input data must be a list of dictionaries'}), 400

        # Chuyển dữ liệu thành DataFrame
        df = pd.DataFrame(data)

        # Kiểm tra dữ liệu đầu vào
        required_fields = ['DayOn', 'Qoil', 'GOR', 'Press_WH', 'LiqRate']
        for col in required_fields:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400

        # Thực hiện dự đoán
        df['Predicted_Oilrate'] = model.predict(df[required_fields])

        # Tạo tệp Excel
        output_file = "predictions.xlsx"
        df.to_excel(output_file, index=False)

        # Gửi tệp về client
        return send_file(output_file, as_attachment=True, download_name="predictions.xlsx")

    except Exception as e:
        return jsonify({'error': str(e)}), 400
    

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render cung cấp biến PORT
    print(f"Running on port {port}")  # Debug
    app.run(host='0.0.0.0', port=port)