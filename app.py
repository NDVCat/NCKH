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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ yêu cầu
        data = request.get_json(force=True)
        
        # Chuyển đổi dữ liệu thành DataFrame
        input_data = pd.DataFrame(data, index=[0])
        
        # Thực hiện dự đoán
        prediction = model.predict(input_data)
        
        # Trả về dự đoán dưới dạng phản hồi JSON
        return jsonify({'Predicted_Oilrate': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)