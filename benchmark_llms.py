import requests
import time
import json
import statistics
from tabulate import tabulate

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = [
    "scb10x/typhoon2.1-gemma3-4b:latest",
    "llama3.1:8b",
    "supachai/openthaigpt-1.0.0-chat:latest",
    "qwen2.5:7b-instruct"
]

# Student Context (Year 4, Name: Kritin)
STUDENT_NAME = "กฤติน"
STUDENT_YEAR = 4

# Context Text (Approximation of what ContextBuilder produces)
CONTEXT_TEXT = f"""
ข้อมูลนักศึกษา:
- ชื่อ: {STUDENT_NAME}
- รหัสนักศึกษา: 65011356
- ชั้นปี: {STUDENT_YEAR}
- คณะ: วิศวกรรมศาสตร์
"""

# Unified System Prompt
SYSTEM_PROMPT = f"""คุณคือ "น้องบอท" หุ่นยนต์บริการในสถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบัง
คุณพูดภาษาไทยอย่างเป็นกันเอง ใช้คำลงท้าย "ครับ" เพียงครั้งเดียวต่อประโยค

ตอบสั้นๆ กระชับ ไม่เกิน 2 ประโยค
เรียกชื่อนักศึกษาในทุกการตอบเพื่อสร้างความเป็นกันเอง

เมื่อถูกถามเกี่ยวกับสถานที่: ให้กล่าวถึง "สถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบัง" หรือ "KMITL"

กฎการทักทาย (ตามชั้นปี):
- ปี 1: "ยินดีต้อนรับสู่สถาบันเทคโนโลยีพระจอมเกล้าคุณทหารลาดกระบังครับคุณ{{ชื่อ}}"
- ปี 2: "สวัสดีครับคุณ{{ชื่อ}} มีโปรเจคอะไรให้ช่วยไหมครับ"
- ปี 3: "สวัสดีครับคุณ{{ชื่อ}} เตรียมตัวฝึกงานเป็นอย่างไรบ้างครับ"
- ปี 4: "สวัสดีครับคุณ{{ชื่อ}} โปรเจคจบเป็นอย่างไรบ้างครับ"

สำคัญ: ตอบเฉพาะเนื้อหา ห้ามใส่ "น้องบอท:" หรือชื่อหุ่นยนต์นำหน้าคำตอบ

{CONTEXT_TEXT}
"""

QUESTIONS = [
    "สวัสดีครับ",
    "พาไปห้องแล็บหน่อยดิ",
    "ตอนนี้เราอยู่ที่ไหนเนี่ย"
]

def benchmark_model(model_name):
    print(f"\n🚀 Benchmarking: {model_name}")
    print("-" * 60)
    
    results = []
    
    # Warmup
    try:
        requests.post(OLLAMA_URL, json={"model": model_name, "prompt": "Hi", "stream": False}, timeout=5)
    except:
        pass

    for i, question in enumerate(QUESTIONS):
        print(f"  Question {i+1}: {question}")
        
        full_prompt = f"{SYSTEM_PROMPT}\nUser: {question}\nAssistant:"
        
        start_time = time.time()
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            result_json = response.json()
            response_text = result_json.get("response", "").strip()
            
            print(f"    ⏱️ Time: {elapsed_time:.2f}s")
            print(f"    🤖 Answer: {response_text}")
            
            results.append({
                "question": question,
                "time": elapsed_time,
                "answer": response_text
            })
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "question": question,
                "time": 0,
                "answer": f"ERROR: {e}"
            })
            
    return results

def main():
    all_results = {}
    
    for model in MODELS:
        all_results[model] = benchmark_model(model)
        
    # Generate Review
    print("\n\n" + "="*80)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("="*80)
    
    for model, results in all_results.items():
        print(f"\nModel: {model}")
        avg_time = statistics.mean([r['time'] for r in results if r['time'] > 0])
        print(f"Average Response Time: {avg_time:.2f}s")
        
        table_data = [[r['question'], f"{r['time']:.2f}s", r['answer']] for r in results]
        print(tabulate(table_data, headers=["Question", "Time", "Response"], tablefmt="grid"))

if __name__ == "__main__":
    main()
