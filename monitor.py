"""
System Monitoring Script
Checks health of all services and provides performance metrics
"""

import requests
import json
import time
import psutil
from datetime import datetime
from typing import Dict, List
import subprocess


class SystemMonitor:
    """Monitor all system components"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.ollama_url = "http://localhost:11434"
        
    def check_api_health(self) -> Dict:
        """Check API server health"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_url}/", timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                return {
                    "status": "✅ HEALTHY",
                    "response_time_ms": f"{response_time:.2f}",
                    "details": response.json()
                }
            else:
                return {
                    "status": "⚠️ DEGRADED",
                    "response_time_ms": f"{response_time:.2f}",
                    "error": f"Status code: {response.status_code}"
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "❌ DOWN",
                "error": "Connection refused - server not running"
            }
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    def check_ollama_health(self) -> Dict:
        """Check Ollama service"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                return {
                    "status": "✅ HEALTHY",
                    "models_loaded": len(model_names),
                    "models": model_names
                }
            else:
                return {
                    "status": "⚠️ DEGRADED",
                    "error": f"Status code: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "❌ DOWN",
                "error": str(e)
            }
    
    def check_docker_health(self) -> Dict:
        """Check Docker and Milvus containers"""
        try:
            # Check Docker
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}:{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return {
                    "status": "❌ ERROR",
                    "error": "Docker command failed"
                }
            
            containers = result.stdout.strip().split('\n')
            milvus_containers = [c for c in containers if 'milvus' in c.lower()]
            
            if not milvus_containers:
                return {
                    "status": "⚠️ WARNING",
                    "message": "No Milvus containers found"
                }
            
            running_count = sum(1 for c in milvus_containers if 'Up' in c)
            
            return {
                "status": "✅ HEALTHY" if running_count > 0 else "❌ DOWN",
                "milvus_containers": len(milvus_containers),
                "running": running_count,
                "containers": milvus_containers
            }
            
        except FileNotFoundError:
            return {
                "status": "❌ ERROR",
                "error": "Docker not found or not in PATH"
            }
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_usage_percent": cpu_percent,
            "memory_used_gb": f"{memory.used / (1024**3):.2f}",
            "memory_total_gb": f"{memory.total / (1024**3):.2f}",
            "memory_percent": memory.percent,
            "disk_used_gb": f"{disk.used / (1024**3):.2f}",
            "disk_total_gb": f"{disk.total / (1024**3):.2f}",
            "disk_percent": disk.percent
        }
    
    def test_memory_operations(self) -> Dict:
        """Test basic memory operations"""
        try:
            # Test insert
            insert_data = {
                "text": f"Monitor test at {datetime.now().isoformat()}",
                "memory_type": "test",
                "student_id": "",
                "timestamp": int(time.time())
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/memory/insert",
                json=insert_data,
                timeout=10
            )
            insert_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return {
                    "status": "❌ FAILED",
                    "operation": "insert",
                    "error": response.text
                }
            
            # Test search
            search_data = {
                "query": "monitor test",
                "top_k": 5
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/memory/search",
                json=search_data,
                timeout=10
            )
            search_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return {
                    "status": "❌ FAILED",
                    "operation": "search",
                    "error": response.text
                }
            
            result = response.json()
            
            return {
                "status": "✅ PASSED",
                "insert_time_ms": f"{insert_time:.2f}",
                "search_time_ms": f"{search_time:.2f}",
                "results_found": result["count"]
            }
            
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    def test_conversation(self) -> Dict:
        """Test basic conversation flow"""
        try:
            session_id = f"monitor_test_{int(time.time())}"
            
            # Update context
            requests.post(
                f"{self.api_url}/context/update",
                json={"session_id": session_id, "location": "test"},
                timeout=5
            )
            
            # Send test message
            speech_data = {
                "session_id": session_id,
                "text": "สวัสดี",
                "confidence": 0.95
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/speech/input",
                json=speech_data,
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return {
                    "status": "❌ FAILED",
                    "error": response.text
                }
            
            result = response.json()
            
            # Cleanup
            requests.delete(f"{self.api_url}/session/{session_id}")
            
            return {
                "status": "✅ PASSED",
                "response_time_ms": f"{response_time:.2f}",
                "response_text": result["response_text"][:50] + "...",
                "intent": result["intent"]
            }
            
        except Exception as e:
            return {
                "status": "❌ ERROR",
                "error": str(e)
            }
    
    def run_full_check(self):
        """Run comprehensive system check"""
        print("="*70)
        print("  ROS2 Robot AI Brain - System Health Monitor")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*70)
        
        # API Health
        print("\n[1] API Server Health")
        print("-" * 70)
        api_health = self.check_api_health()
        for key, value in api_health.items():
            if key != "details":
                print(f"  {key}: {value}")
        
        if api_health["status"] != "✅ HEALTHY":
            print("\n⚠️ API server is not healthy. Skipping further tests.")
            return
        
        # Ollama Health
        print("\n[2] Ollama LLM Service")
        print("-" * 70)
        ollama_health = self.check_ollama_health()
        for key, value in ollama_health.items():
            if key != "models":
                print(f"  {key}: {value}")
            elif isinstance(value, list):
                for model in value:
                    print(f"    - {model}")
        
        # Docker/Milvus Health
        print("\n[3] Docker & Milvus")
        print("-" * 70)
        docker_health = self.check_docker_health()
        for key, value in docker_health.items():
            if key != "containers":
                print(f"  {key}: {value}")
        
        # System Metrics
        print("\n[4] System Resources")
        print("-" * 70)
        metrics = self.get_system_metrics()
        print(f"  CPU Usage: {metrics['cpu_usage_percent']}%")
        print(f"  Memory: {metrics['memory_used_gb']} GB / {metrics['memory_total_gb']} GB ({metrics['memory_percent']}%)")
        print(f"  Disk: {metrics['disk_used_gb']} GB / {metrics['disk_total_gb']} GB ({metrics['disk_percent']}%)")
        
        # Performance warning
        if metrics['cpu_usage_percent'] > 80:
            print("  ⚠️ High CPU usage detected")
        if metrics['memory_percent'] > 85:
            print("  ⚠️ High memory usage detected")
        if metrics['disk_percent'] > 90:
            print("  ⚠️ Low disk space")
        
        # Functional Tests
        print("\n[5] Memory Operations Test")
        print("-" * 70)
        memory_test = self.test_memory_operations()
        for key, value in memory_test.items():
            print(f"  {key}: {value}")
        
        print("\n[6] Conversation Flow Test")
        print("-" * 70)
        conv_test = self.test_conversation()
        for key, value in conv_test.items():
            print(f"  {key}: {value}")
        
        # Summary
        print("\n" + "="*70)
        print("  Summary")
        print("="*70)
        
        all_checks = [
            ("API Server", api_health["status"]),
            ("Ollama LLM", ollama_health["status"]),
            ("Docker/Milvus", docker_health["status"]),
            ("Memory Ops", memory_test["status"]),
            ("Conversation", conv_test["status"])
        ]
        
        passed = sum(1 for _, status in all_checks if "✅" in status)
        total = len(all_checks)
        
        for name, status in all_checks:
            print(f"  {status} {name}")
        
        print(f"\n  Overall: {passed}/{total} checks passed")
        
        if passed == total:
            print("\n  ✅ All systems operational!")
        else:
            print(f"\n  ⚠️ {total - passed} system(s) need attention")
        
        print("="*70)


def continuous_monitor(interval_seconds: int = 60):
    """Run continuous monitoring"""
    monitor = SystemMonitor()
    
    print("Starting continuous monitoring...")
    print(f"Checking every {interval_seconds} seconds. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            monitor.run_full_check()
            print(f"\nNext check in {interval_seconds} seconds...\n")
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


if __name__ == "__main__":
    import sys
    
    monitor = SystemMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        continuous_monitor(interval)
    else:
        # Single check
        monitor.run_full_check()
        
        print("\n💡 Tip: Run with --continuous [seconds] for continuous monitoring")
        print("   Example: python monitor.py --continuous 30")
