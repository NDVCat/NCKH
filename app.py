import joblib
from flask import Flask, request, jsonify
import pandas as pd
import os

# Giả sử bạn đã train mô hình với 5 cột: 
# ["DayOn", "Qoil", "GOR", "Press_WH", "LiqRate"]
# và lưu thành 'oilrate_model.pkl'

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
        # 1) Lấy dữ liệu
        if request.method == 'POST':
            data = request.get_json(force=True)
        else:  # GET
            data = request.args.to_dict()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # 2) Kiểm tra và ép kiểu cho 5 cột
        #    Cần tất cả key ["DayOn", "Qoil", "GOR", "Press_WH", "LiqRate"].
        try:
            # Tạo dict mới chỉ chứa 5 key cần thiết
            required_columns = ["DayOn", "Qoil", "GOR", "Press_WH", "LiqRate"]
            
            # Kiểm tra thiếu cột
            missing_cols = [col for col in required_columns if col not in data]
            if missing_cols:
                return jsonify({
                    'error': f'Missing required columns: {missing_cols}'
                }), 400

            # Ép kiểu float (hoặc int) cho từng cột
            # Nếu cột nào không parse được -> ValueError
            input_dict = {
                col: float(data[col]) for col in required_columns
            }
        except ValueError:
            return jsonify({
                'error': 'Invalid input format. All values must be numeric'
            }), 400

        print("Dữ liệu nhận được:", input_dict)  # Debug

        # 3) Tạo DataFrame đúng thứ tự cột
        input_data = pd.DataFrame([input_dict], columns=required_columns)

        # 4) Dự đoán
        prediction = model.predict(input_data)

        # 5) Trả về kết quả
        return jsonify({'Predicted_Oilrate': float(prediction[0])})

    except Exception as e:
        print("Lỗi:", str(e))
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Render sẽ cung cấp biến môi trường PORT, nếu không có thì mặc định 10000
    port = int(os.environ.get("PORT", 10000))
    print(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port)