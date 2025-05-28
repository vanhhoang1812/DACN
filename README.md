# Chatbot Áo Dài - Dự án sử dụng TensorFlow và Flask

## 🧠 Mô tả

Đây là một chatbot hỗ trợ tư vấn về áo dài, được xây dựng bằng Python, TensorFlow, Flask và sử dụng embedding từ Wikipedia tiếng Việt.

---

## 🚀 Hướng dẫn cài đặt

### 1. Cài đặt TensorFlow

Vui lòng làm theo hướng dẫn chính thức từ TensorFlow tại:

🔗 https://www.tensorflow.org/install/pip?hl=vi

Ví dụ (với Python 3.8+ và pip):
```
pip install tensorflow
```

### 2. Cài đặt các thư viện cần thiết
Tất cả các thư viện phụ thuộc đã được liệt kê trong file requirements.txt. Cài đặt bằng lệnh:

```
pip install -r requirements.txt
```

### 3. Tải embedding wiki.vi.vec
Tải tệp wiki.vi.vec tại liên kết sau:

🔗 https://www.kaggle.com/datasets/vanhhong/vector-corpus-wiki

Sau khi tải về, vui lòng đặt file vào thư mục embedding/ trong dự án:
```
project/
├── embedding/
│   └── wiki.vi.vec
```

### ▶️ Chạy chương trình
Sử dụng lệnh sau để chạy ứng dụng Flask:
```
flask run
```
Ứng dụng sẽ khởi động tại địa chỉ: http://127.0.0.1:5000
