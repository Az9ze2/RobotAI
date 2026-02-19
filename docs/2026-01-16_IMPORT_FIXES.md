# Import Path Fixes - 2026-01-16

## Overview

After restructuring the project to follow ML best practices, all Python import paths have been updated to work with the new directory structure.

## Problem

The project restructure moved source code from root directory to `src/` directory, requiring all import paths to be updated.

## Changes Made

### 1. Entrypoint Files

All entrypoint scripts now correctly reference `src/` directory:

**Files Fixed:**
- `entrypoint/demo_voice.py`
- `entrypoint/inference.py`
- `entrypoint/train.py`

**Change:**
```python
# OLD (incorrect after restructure)
sys.path.insert(0, str(Path(__file__).parent))

# NEW (correct)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

### 2. Test Files

All test scripts updated to reference `src/`:

**Files Fixed:**
- `tests/test_voice_pipeline.py`
- `tests/test_stt_simple.py`

**Change:**
```python
# OLD
sys.path.insert(0, str(Path(__file__).parent))

# NEW
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

### 3. API Main File

API server updated for correct config path and imports:

**File:** `src/api/main.py`

**Changes:**
1. Import path fix:
```python
# OLD
sys.path.append(str(Path(__file__).parent.parent))

# NEW
sys.path.insert(0, str(Path(__file__).parent.parent))
```

2. Config path fix:
```python
# OLD
config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

# NEW (goes up one more level from src/api/)
config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
```

### 4. Pipeline Files

**File:** `src/pipelines/inference_pipeline.py`

**Change:**
```python
# OLD (incorrect module name)
from llm.ollama_client import OllamaClient

# NEW (correct module name)
from llm.typhoon_client import TyphoonClient
```

Also updated instantiation:
```python
# OLD
self.llm = OllamaClient(
    model_name=config['llm']['model'],
    api_url=config['llm']['api_url']
)

# NEW (correct parameter names)
self.llm = TyphoonClient(
    model=config['llm']['model'],
    api_url=config['llm']['api_url']
)
```

### 5. Config Path Fixes

**Files with config path updates:**
- `entrypoint/inference.py`
- `entrypoint/train.py`

**Change:**
```python
# OLD (incorrect - relative path)
with open("config/settings.yaml", "r", encoding="utf-8") as f:
    self.config = yaml.safe_load(f)

# NEW (correct - absolute path from script location)
config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    self.config = yaml.safe_load(f)
```

## Directory Structure Reference

```
RobotAI/
├── config/
│   └── settings.yaml          # Config file location
├── entrypoint/                # Entry points need: parent.parent / 'src'
│   ├── demo_voice.py
│   ├── inference.py
│   └── train.py
├── src/                       # Source code base
│   ├── api/                   # API needs: parent.parent (to src/)
│   │   └── main.py           # Config needs: parent.parent.parent (to RobotAI/)
│   ├── stt/
│   ├── tts/
│   ├── llm/
│   │   └── typhoon_client.py  # Correct module name
│   ├── pipelines/
│   │   └── inference_pipeline.py
│   └── utils/
└── tests/                     # Tests need: parent.parent / 'src'
    ├── test_voice_pipeline.py
    └── test_stt_simple.py
```

## Path Calculation Guide

From any file, to access `src/`:

| Current Location | Path to src/ | Example |
|-----------------|-------------|---------|
| `entrypoint/` | `parent.parent / 'src'` | `entrypoint/inference.py` |
| `tests/` | `parent.parent / 'src'` | `tests/test_*.py` |
| `src/api/` | `parent.parent` | `src/api/main.py` |
| `src/pipelines/` | `parent.parent` | `src/pipelines/*.py` |

To access `config/settings.yaml`:

| Current Location | Path to config/ |
|-----------------|----------------|
| `entrypoint/` | `parent.parent / 'config'` |
| `src/api/` | `parent.parent.parent / 'config'` |
| `src/pipelines/` | `parent.parent.parent / 'config'` |

## Verification

### Test Imports Work

Run from project root:

```bash
# Test entrypoint imports
python entrypoint/inference.py

# Test standalone test
python tests/test_voice_pipeline.py

# Test API
python src/api/main.py
```

### Common Import Pattern

All files now follow this pattern:

```python
import sys
from pathlib import Path

# Add src/ to path (adjust based on location)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Now can import from src/ modules
from stt.whisper_client import WhisperSTT
from tts.vachana_client import VachanaTTS
from llm.typhoon_client import TyphoonClient
```

## Files Modified

| File | Change Type |
|------|-------------|
| `entrypoint/demo_voice.py` | Import path |
| `entrypoint/inference.py` | Import path + config path |
| `entrypoint/train.py` | Import path + config path |
| `tests/test_voice_pipeline.py` | Import path |
| `tests/test_stt_simple.py` | Import path |
| `src/api/main.py` | Import path + config path |
| `src/pipelines/inference_pipeline.py` | Module name + parameter names |

**Total Files Modified:** 7

## Breaking Changes Fixed

1. ❌ **OLD:** `from llm.ollama_client import OllamaClient` 
   ✅ **NEW:** `from llm.typhoon_client import TyphoonClient`

2. ❌ **OLD:** Relative config paths (`"config/settings.yaml"`)
   ✅ **NEW:** Absolute paths from file location

3. ❌ **OLD:** `sys.path.append` in some files
   ✅ **NEW:** `sys.path.insert(0, ...)` everywhere (higher priority)

## Testing Checklist

After these fixes, the following should work:

- ✅ `python entrypoint/inference.py` - Main voice chat
- ✅ `python entrypoint/demo_voice.py` - Voice demo
- ✅ `python entrypoint/train.py` - Training entrypoint
- ✅ `python tests/test_voice_pipeline.py` - Pipeline tests
- ✅ `python tests/test_stt_simple.py` - STT tests
- ✅ `python src/api/main.py` - API server
- ✅ All imports resolve correctly from any entry point

## Additional Notes

### VachanaTTS External Path

The `src/tts/vachana_client.py` has a hardcoded external path:

```python
VACHANA_PATH = Path("C:/Users/Win 10 Pro/Desktop/VachanaTTS")
```

This is **intentional** because VachanaTTS is an external dependency, not part of RobotAI project structure.

### No Changes Needed

These files don't need sys.path modifications because they're modules imported by others, not entry points:
- `src/stt/whisper_client.py`
- `src/tts/vachana_client.py` (has external path only)
- `src/llm/typhoon_client.py`
- `src/utils/audio_utils.py`
- `src/utils.py`

## Result

✅ **All import paths are now correct and consistent with the new ML project structure.**

All entry points can correctly locate and import from `src/` modules, and all config file paths resolve correctly regardless of where scripts are run from.

---

**Date:** 2026-01-16  
**Status:** ✅ Complete  
**Impact:** All Python files now work with new directory structure
