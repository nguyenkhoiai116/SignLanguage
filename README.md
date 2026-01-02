# Sign Language Recognition using ResNet18 🤟

Dự án nhận diện ngôn ngữ ký hiệu (Sign Language) sử dụng mô hình **ResNet18** với kỹ thuật **Transfer Learning** và **Fine-tuning**.

Project được xây dựng bằng **PyTorch**, hỗ trợ training trên GPU, tự động lưu model tốt nhất và vẽ biểu đồ đánh giá.

## 📂 Cấu trúc thư mục 
Để code chạy được, bạn cần sắp xếp dữ liệu theo cấu trúc chuẩn xem mẫu ở file tree.txt
## ⚙️ Cài đặt

1. **Clone repo:**
   ```bash
   git clone [https://github.com/nguyenkhoiai116/SignLanguge.git](https://github.com/nguyenkhoiai116/SignLanguge.git)
   cd SignLanguge

2. **Cài thư viện**
pip install -r requirements.txt

3. **Huấn luyện**
python train.py
Code sẽ tự động tải ResNet18 pre-trained.

Bắt đầu training 30 epochs.

Model tốt nhất sẽ được lưu thành best_model_checkpoint.pth.

Model cuối cùng được lưu thành sign_language_resnet18_finetune.pth.

Biểu đồ huấn luyện được lưu thành hình ảnh training_curves.png.
4. **🛠 Công nghệ sử dụng**
Python 3.10.9

PyTorch & Torchvision

Matplotlib (Vẽ biểu đồ)

Tqdm (Thanh tiến trình)