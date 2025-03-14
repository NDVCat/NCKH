import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os

# Tải mô hình đã huấn luyện
try:
    model = joblib.load('oilrate_model.pkl')
    print("Model loaded successfully")
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

        # Kiểm tra và ép kiểu dữ liệu
        try:
            data = {key: float(value) for key, value in data.items() if value is not None and value != ""}
        except ValueError:
            return jsonify({'error': 'Invalid input format. All values must be numeric'}), 400

        print("Dữ liệu nhận được:", data)  # Debugging

        # Chuyển đổi dữ liệu thành DataFrame
        input_data = pd.DataFrame([data])

        # Thực hiện dự đoán
        prediction = model.predict(input_data)

        # Tạo tệp Excel
        output_file = "predictions.xlsx"
        df.to_excel(output_file, index=False)

        # Gửi tệp về client
        return send_file(output_file, as_attachment=True, download_name="predictions.xlsx")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render cung cấp biến PORT
    print(f"Running on port {port}")  # Debug
    app.run(host='0.0.0.0', port=port)