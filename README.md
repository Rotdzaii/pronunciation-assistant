# Pronunciation Assistant AI

Tai lieu nay huong dan chi tiet de demo du an tren may khac (fresh machine), tu cai dat moi truong den khoi dong backend va mo giao dien.

## 1) Tong quan nhanh

- Backend: FastAPI (Python)
- AI core: WhisperX + G2P + Prosody (Librosa + Parselmouth)
- UI demo: Web tinh tai static/index.html (duoc phuc vu qua FastAPI)
- Database cloud: Supabase (tuy chon, co the bo qua khi demo local)

## 2) Yeu cau he thong

Khuyen nghi toi thieu:

- Windows 10/11 (du an da co script run_server.bat cho Windows)
- Python 3.12.x
- Git
- FFmpeg (de xu ly audio cho mot so thu vien)
- Ket noi Internet de tai dependencies va model

Khuyen nghi de chay nhanh:

- GPU NVIDIA (vi du RTX 3050) + driver moi
- CUDA phu hop voi ban torch trong requirements (torch cu121)

Ghi chu:

- Co the chay CPU neu khong co GPU, nhung se cham hon.
- Lan dau chay se mat them thoi gian de tai model/du lieu phu tro.

## 3) Clone source code

```powershell
git clone <REPO_URL>
cd pronunciation-assistant
```

Neu da co source code san (file zip), chi can giai nen va mo dung thu muc goc du an.

## 4) Tao va kich hoat moi truong ao

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Neu bi chan script execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau do mo lai PowerShell va activate lai .venv.

## 5) Cai dat dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Cai FFmpeg neu may chua co:

```powershell
winget install Gyan.FFmpeg
```

Kiem tra nhanh:

```powershell
python --version
ffmpeg -version
```

Luu y:

- Buoc nay co the mat kha lau do co nhieu goi AI.
- WhisperX duoc cai truc tiep tu GitHub theo requirements.

## 6) Tao file .env

Tao file .env o thu muc goc du an voi noi dung:

```env
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-anon-or-service-role-key>
```

Neu chua co Supabase, app van chay local demo duoc, nhung phan luu cloud se bao canh bao.

## 7) Chay du an

### Cach 1 (khuyen nghi tren Windows): dung script san co

```powershell
run_server.bat
```

Script se:

- Kiem tra thu muc .venv
- Kiem tra CUDA status
- Khoi dong FastAPI tai cong 8000

### Cach 2 (thu cong)

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

## 8) Truy cap giao dien demo

Mo trinh duyet:

```text
http://127.0.0.1:8000
```

Sau do:

1. Nhan nut BAT DAU NOI.
2. Cho phep trinh duyet truy cap microphone.
3. Noi mot cau tieng Anh ro rang.
4. Nhan DUNG & PHAN TICH.
5. Xem ket qua STT, thinking pipeline, phoneme details, overall score.

## 9) Kiem tra nhanh endpoint

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Endpoint chinh:

- GET /
- POST /upload-audio

## 10) Loi thuong gap va cach xu ly

1. Khong mo duoc mic tren browser
- Kiem tra quyen microphone cua browser va he dieu hanh.
- Thu refresh page, dong/mo lai tab.

2. Port 8000 dang bi chiem
- Doi cong:
	- uvicorn api.main:app --reload --port 8001
- Hoac tat tien trinh dang dung cong 8000.

3. CUDA status la False
- App van chay duoc bang CPU.
- Kiem tra lai NVIDIA driver/CUDA va ban torch.

4. Bao loi thieu module
- Dam bao dang activate .venv.
- Chay lai pip install -r requirements.txt.

5. Supabase loi ket noi
- Kiem tra SUPABASE_URL va SUPABASE_KEY trong .env.
- Neu demo local, co the tiep tuc bo qua buoc luu cloud.

## 11) Cac file quan trong de demo

- api/main.py: FastAPI app va endpoint upload audio
- core/phoneme_engine.py: STT + phoneme pipeline hien tai
- core/prosody_engine.py: phan tich pitch/intensity/duration
- core/database.py: ghi ket qua len Supabase
- static/index.html: giao dien demo va xu ly ghi am tren browser
- run_server.bat: script khoi dong nhanh tren Windows

## 12) Demo checklist (5 phut)

1. Da tao .venv va cai dependencies.
2. Da tao .env (neu can luu cloud).
3. Da chay run_server.bat thanh cong.
4. Mo duoc trang http://127.0.0.1:8000.
5. Ghi am va nhan duoc ket qua phan tich.
6. (Tuy chon) Kiem tra logs Supabase co ban ghi moi.

---

Neu ban muon, co the bo sung them phien ban README cho Linux/macOS va script setup tu dong 1 lenh (bootstrap) de doi demo khoi dong nhanh hon.
