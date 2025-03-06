{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f51cf92a-e9d0-4190-b672-fe81d1d1a8f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "24c1a767-85e3-46eb-8426-756ccc4475f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tải mô hình đã huấn luyện\n",
    "try:\n",
    "    model = joblib.load('oilrate_model.pkl')\n",
    "except Exception as e:\n",
    "    print(f\"Lỗi khi tải mô hình: {e}\")\n",
    "    exit(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "3d051ebc-930e-47e2-b33d-056a9f018860",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, request, jsonify\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9edccba9-6113-4479-a78b-b8b8817fdea2",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = Flask(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "2774e722-a004-42d1-9dba-8da558277879",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    try:\n",
    "        # Lấy dữ liệu từ yêu cầu\n",
    "        data = request.get_json(force=True)\n",
    "        \n",
    "        # Chuyển đổi dữ liệu thành DataFrame\n",
    "        input_data = pd.DataFrame(data, index=[0])\n",
    "        \n",
    "        # Thực hiện dự đoán\n",
    "        prediction = model.predict(input_data)\n",
    "        \n",
    "        # Trả về dự đoán dưới dạng phản hồi JSON\n",
    "        return jsonify({'Predicted_Oilrate': prediction[0]})\n",
    "    except Exception as e:\n",
    "        return jsonify({'error': str(e)}), 400"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "5cceef5f-8482-45b5-8d75-02b790ea1c05",
   "metadata": {},
   "outputs": [],
   "source": [
    "if __name__ == '__main__':\n",
    "    app.run(host='0.0.0.0', port=5005)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "ace03af8-713e-4c65-91a6-574482f2305e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
