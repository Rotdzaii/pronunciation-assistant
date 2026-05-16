import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Khởi tạo Supabase Client
if not url or not key:
    print("--- ⚠️ CẢNH BÁO: Thiếu cấu hình Supabase trong .env ---")
    supabase: Client = None
else:
    supabase: Client = create_client(url, key)

def save_attempt_to_cloud(target_word: str, score: float, detailed_analysis: list):
    if not supabase:
        print("--- ❌ LỖI: Chưa kết nối được Supabase ---")
        return None

    try:
        # ĐÃ CẬP NHẬT TÊN CỘT KHỚP VỚI ẢNH CẬU GỬI
        data = {
            "target_word": target_word,
            "overall_score": round(score, 2),  # Khớp với cột 'overall_score'
            "phoneme_details": detailed_analysis, # Khớp với cột 'phoneme_details'
            "created_at": "now()"
        }
        
        # Đảm bảo tên bảng trên Supabase của cậu là 'practice_history'
        response = supabase.table("practice_history").insert(data).execute()
        print(f"--- ✅ Đã lưu kết quả từ '{target_word}' lên Cloud! ---")
        return response
    except Exception as e:
        print(f"--- ❌ LỖI DATABASE: {str(e)} ---")
        return None