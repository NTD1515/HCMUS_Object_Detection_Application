# 🎯 HCMUS Object Detection Application

Đồ án cuối kỳ môn Học thống kê, trường Đại học Khoa học Tự nhiên, ĐHQG-HCM (HCMUS). 

## 👥 Thành viên thực hiện
* **Nguyễn Trung Dũng** - MSSV: 21120228
* **Sần Dịch Anh** - MSSV: 21120411

## 📝 Giới thiệu đồ án
Đồ án tập trung vào việc xây dựng và so sánh các kiến trúc học sâu tiên tiến nhất cho bài toán Phát hiện vật thể (Object Detection) trên tập dữ liệu CheXpert. 


Các mô hình được triển khai và đánh giá trong dự án bao gồm:
* **YOLO** (You Only Look Once)
* **Faster R-CNN**
* **Transformer-based Object Detection** (RT-DETR)

Toàn bộ mô hình đã được tích hợp thành một ứng dụng Web tương tác trực quan để dễ dàng demo nghiệm thu.

---

## 🖥️ Môi trường Huấn luyện (Training)
* **Phần cứng:** Toàn bộ các mô hình được huấn luyện hoàn toàn trên môi trường **Google Colab**, sử dụng GPU **Tesla T4**.
* **Kết quả đánh giá:** Biểu đồ huấn luyện (Loss/Epoch), các chỉ số đánh giá (mAP, Precision, Recall) và trọng số (weights) được đính kèm chi tiết bên trong từng thư mục của mỗi bộ mô hình (`FasterRCNN_Runs`, `Transformer_Runs`, `YOLO_Runs`).

---

## 🗂️ Cấu trúc thư mục (Project Tree)
Dự án được tổ chức theo cấu trúc sau để phân tách rõ ràng giữa code huấn luyện, dữ liệu và mã nguồn giao diện:

```text
HCMUS_Object_Detection_Application/
├── Data                        # Thư mục chứa các bộ dữ liệu, xem file link data.txt để tải về, bỏ các thư mục vào thư mục Data là ổn
|   ├── FasterRCNN_Runs/            # Thư mục chứa kết quả đánh giá và biểu đồ của Faster R-CNN
|   ├── Transformer_Runs/           # Thư mục chứa kết quả đánh giá và biểu đồ của Transformer
|   └── YOLO_Runs/                  # Thư mục chứa kết quả đánh giá và biểu đồ của YOLO
├── app.py                      # Mã nguồn chạy giao diện Web Application (Streamlit)
├── HTK_offline.ipynb           # Notebook chứa code để chạy trên máy tính cá nhân (Local Machine)
├── HTK_online.ipynb            # Notebook cấu hình sẵn đường dẫn để nạp và chạy trên Google Colab
├── link data.txt               # Chứa đường dẫn tải Dataset và file Trọng số (Weights) kích thước lớn
├── requirements.txt            # Danh sách các thư viện Python cần thiết
├── .gitignore                  # File cấu hình bỏ qua các tệp tạm và tệp dữ liệu lớn khi đẩy lên Git

```
## 💾 Dữ liệu (Dataset & Weights)
Do giới hạn dung lượng của GitHub, bộ dữ liệu hình ảnh gốc và các file trọng số lớn của mô hình (.pt, .pth) không được đẩy trực tiếp lên kho lưu trữ này.

👉 Vui lòng mở file link data.txt để lấy đường dẫn Google Drive. Bạn cần tải về và đặt chúng vào đúng cấu trúc thư mục tương ứng trước khi chạy thử nghiệm.

## 🚀 Hướng dẫn Cài đặt & Sử dụng
1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python (khuyến nghị phiên bản 3.9+). Mở Terminal/Command Prompt tại thư mục gốc của dự án và chạy lệnh sau để tải các gói thư viện cần thiết:
```
Bash
pip install -r requirements.txt
```

2. Thử nghiệm trên Jupyter Notebook
Dự án cung cấp sẵn 2 phiên bản mã nguồn để bạn có thể xem lại quá trình huấn luyện hoặc tự test mã:

Dùng trên máy cá nhân (Local): Mở file HTK_offline.ipynb (đường dẫn đọc file đã được tinh chỉnh cho máy tính cục bộ).

Dùng trên Google Colab: Upload file HTK_online.ipynb lên Google Colab, mount Drive chứa dataset và ấn chạy tuần tự các cell.

3. Chạy Giao diện Web (Web App)
Để khởi chạy ứng dụng giao diện nhận diện vật thể trực tiếp trên trình duyệt, hãy chạy lệnh sau trong Terminal:
```
Bash
streamlit run app.py
```

Hệ thống sẽ tự động cung cấp một đường dẫn Local URL (thường là http://localhost:8501) để bạn tải ảnh lên và xem mô hình dự đoán.
