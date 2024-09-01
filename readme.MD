# Quy trình chạy train server AIoT Lab VN

- Liên hệ: Đức Đạt (Zalo: 0902850103)

## 1. Mục đích sử dụng
- Học tập, nghiên cứu tại AIoT Lab VN
- Chạy train model AI, ML, LLM 

## 2. Quy trình và cấu trúc source code 
- Bước 1: Đẩy source lên github (có thể để private) và add quyền Collaborators account [ducdatit2002](https://www.github.com/ducdatit2002)

- Bước 2: Source code (như mẫu) bao gồm các file sau: 
(Tham khảo cấu trúc source sau)

```requirements.txt```
(Gồm các thư viện cần thiết và phiên bản để tải các thư viện)
```bash 
torch
ultralytics
numpy
opencv-python-headless==4.8.0.76
scipy==1.12.0
filterpy==1.4.5
matplotlib==3.8.3
```

```main.py``` 
(File để chạy code)

Lưu ý: Code phải được viết để chạy không cần tương tác trực tiếp và có thể thực thi từ đầu đến cuối mà không cần sự can thiệp.

```Readme.MD```
(File hướng dẫn sử dụng code, bao gồm các bước cài đặt môi trường, cách chạy các script, và mô tả các tham số, vị trí folder data.)
```bash
## Cài đặt thư viện
pip install -r requirements.txt

## Chạy train
python3 main.py
```

## 3. Data cho model training

- Data không có file trên 100MB thì đẩy toàn bộ lên github

- Data nặng thì nén file data và đẩy lên GG Drive (vui lòng nén để khi download xuống data dễ bị lẫn lộn)

## 4. Một số lưu ý:
- Tuỳ phù thuộc vào lưu lượng task đang chạy, thời gian trả kết quả train là từ 3 - 7 ngày.

- Cung cấp một cấu trúc thư mục rõ ràng và logic cho dữ liệu. Ví dụ: tách dữ liệu thành các thư mục train, validation, và test.

- Cung cấp thông tin đầy đủ về kiến trúc mô hình, tham số, và bất kỳ thông số cấu hình đặc biệt nào cho việc train mô hình với hình ảnh.
Starting setup...
Initialized empty Git repository in /Volumes/Workspace /AIoT-Lab-VN/server/training-server/.git/
Initialized
Added files
[main (root-commit) 7e67853] Initial commit
 Committer: Pham Duc Dat <phamducdat@10.212.134.200>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 10 files changed, 144 insertions(+)
 create mode 100644 .DS_Store
 create mode 100644 data/.DS_Store
 create mode 100644 data/plant.yaml
 create mode 100644 data/plant/.DS_Store
 create mode 100644 data/plant/test/.DS_Store
 create mode 100644 data/plant/train/.DS_Store
 create mode 100644 data/plant/valid/.DS_Store
 create mode 100644 main.py
 create mode 100644 readme.MD
 create mode 100644 requirements.txt
Committed
Branch renamed to main
Remote added
Pushed to remote
Starting setup...
Reinitialized existing Git repository in /Volumes/Workspace /AIoT-Lab-VN/server/training-server/.git/
Initialized
Added files
[main 113985d] Initial commit
 Committer: Pham Duc Dat <phamducdat@10.212.134.200>
Your name and email address were configured automatically based
on your username and hostname. Please check that they are accurate.
You can suppress this message by setting them explicitly:

    git config --global user.name "Your Name"
    git config --global user.email you@example.com

After doing this, you may fix the identity used for this commit with:

    git commit --amend --reset-author

 1 file changed, 32 insertions(+)
Committed
Branch renamed to main
Remote added
branch 'main' set up to track 'origin/main'.
Pushed to remote
