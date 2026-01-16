"""
System Validation Script
Checks if all components are properly installed and configured
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check_python_version():
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version OK")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def check_dependencies():
    print_section("Python Dependencies")
    required = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
        "pymilvus",
        "sentence_transformers",
        "torch",
        "yaml",
        "loguru"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package if package != "yaml" else "yaml")
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_ollama():
    print_section("Ollama Service")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running")
            print(f"   Installed models: {len(models)}")
            for model in models:
                print(f"   - {model['name']}")
            
            # Check for Typhoon
            has_typhoon = any("typhoon" in m["name"].lower() for m in models)
            if not has_typhoon:
                print("\n⚠️  Typhoon model not found!")
                print("   Run: ollama pull typhoon:7b-instruct")
            return True, has_typhoon
        else:
            print("❌ Ollama service not responding properly")
            return False, False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is installed and running")
        return False, False

def check_docker():
    print_section("Docker & Milvus")
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout
            milvus_running = "milvus" in output.lower()
            etcd_running = "etcd" in output.lower()
            minio_running = "minio" in output.lower()
            
            if milvus_running and etcd_running and minio_running:
                print("✅ Docker is running")
                print("✅ Milvus containers are running")
                return True
            else:
                print("✅ Docker is running")
                print("❌ Milvus containers are NOT running")
                print("\n   Start Milvus with:")
                print("   docker-compose up -d")
                return False
        else:
            print("❌ Docker is not running or not installed")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")
        return False

def check_file_structure():
    print_section("File Structure")
    
    files_to_check = [
        ("api/main.py", True),
        ("config/settings.yaml", True),
        ("llm/typhoon_client.py", True),
        ("mcp/context_builder.py", True),
        ("vector_db/milvus_client.py", False),  # Missing .py extension
        ("docker-compose.yml", True),
        ("requirements.txt", True)
    ]
    
    all_good = True
    for file_path, expected_ok in files_to_check:
        path = Path(file_path)
        if path.exists():
            if expected_ok:
                print(f"✅ {file_path}")
            else:
                print(f"⚠️  {file_path} - File exists but may need fixing")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_good = False
    
    # Check for missing .py extension
    milvus_file = Path("vector_db/milvus_client")
    if milvus_file.exists() and not milvus_file.suffix:
        print("\n⚠️  vector_db/milvus_client is missing .py extension")
        print("   This file should be renamed to milvus_client.py")
        all_good = False
    
    return all_good

def check_directories():
    print_section("Directory Structure")
    
    dirs = ["api", "config", "llm", "mcp", "vector_db", "database", "stt", "tts", "logs"]
    
    for dir_name in dirs:
        path = Path(dir_name)
        if path.exists() and path.is_dir():
            files = list(path.glob("*.py"))
            print(f"✅ {dir_name}/ ({len(files)} Python files)")
        else:
            print(f"❌ {dir_name}/ - NOT FOUND")

def main():
    print("\n" + "="*60)
    print("  ROS2 Robot AI Brain - System Validation")
    print("="*60)
    
    results = {
        "python": check_python_version(),
        "deps": check_dependencies()[0],
        "ollama": check_ollama()[0],
        "docker": check_docker(),
        "files": check_file_structure(),
    }
    
    check_directories()
    
    # Summary
    print_section("Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for check, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check.upper()}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed < total:
        print("\n⚠️  Some components need attention. See details above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start Milvus: docker-compose up -d")
        print("3. Install Typhoon: ollama pull typhoon:7b-instruct")
        print("4. Rename vector_db/milvus_client to vector_db/milvus_client.py")
    else:
        print("\n✅ All systems ready!")

if __name__ == "__main__":
    main()
