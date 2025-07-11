# 📁 Configuration Guide

## 📍 **Authoritative Configuration Location**

**Primary configuration file:** `/home/gta/Work/agent-pa/backend/config/settings.yaml`

This is the **single source of truth** for all AI assistant configuration.

## 🚀 **Quick Start**

1. **Copy the example:**
   ```bash
   cp backend/config/settings.example.yaml backend/config/settings.yaml
   ```

2. **Edit your settings:**
   ```bash
   nano backend/config/settings.yaml
   ```

3. **Add your API keys:**
   ```yaml
   # In settings.yaml, add your keys to .env file:
   GEMINI_API_KEY=your_gemini_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```

4. **Launch the system:**
   ```bash
   python run_dashboard.py
   ```

## 📂 **File Structure**

```
agent-pa/
├── .env                      ← 🔑 API keys (create this)
├── backend/
│   ├── config/
│   │   ├── settings.yaml        ← 🎯 MAIN CONFIG (use this!)
│   │   └── settings.example.yaml ← 📝 Template/documentation
│   └── ...
└── ...
```

## ⚙️ **Configuration Priority**

The system searches for `settings.yaml` in this order:

1. **`backend/config/settings.yaml`** ← 🎯 **Primary (your preferred location)**
2. Project root directory (for compatibility)
3. Current working directory
4. User home directory

## 🎛️ **What You Can Configure**

### **🤖 AI Models**
```yaml
models:
  hierarchy: ["ollama", "openrouter", "gemini"]
  ollama:
    models:
      fast: "llama3.2:3b"
```

### **🤝 Multi-Agent System**
```yaml
autogen:
  agents:
    ceo:
      model: "strategic"
      system_message: "You are the CEO..."
```

### **💰 Cost Management**
```yaml
cost:
  daily_budget_usd: 5.0
  model_costs:
    "openrouter/claude-3-haiku": 0.00025
```

### **🧠 Reasoning Models**
```yaml
reasoning:
  chat_model: "openrouter/anthropic/claude-3-haiku"
  reflection_model: "openrouter/meta-llama/llama-3.1-8b-instruct"
```

## 🔍 **Troubleshooting**

### **Config Not Loading?**
```bash
# Check which config file is being used
python -c "
from backend.core.config import AssistantConfig
config = AssistantConfig()
print('Config loaded from:', config._find_config_file('settings.yaml'))
"
```

### **Multiple Config Files?**
- ✅ Use: `settings.yaml` in project root
- ❌ Ignore: `backend/config/settings.yaml` (old format)

### **API Keys Not Working?**
- ✅ Put API keys in `.env` file (project root)
- ✅ Or set environment variables
- ❌ Don't put API keys directly in `settings.yaml`

## 📚 **Advanced Configuration**

See `settings.example.yaml` for complete documentation of all available options.

## 🔧 **Development**

When adding new configuration options:

1. **Add to `settings.example.yaml`** with documentation
2. **Update `backend/core/config.py`** to handle the new option
3. **Test with both missing and present configurations**

## 🎯 **Best Practices**

- ✅ **Use project root** `settings.yaml` as primary config
- ✅ **Keep API keys** in `.env` file
- ✅ **Document new options** in `settings.example.yaml`
- ✅ **Use environment-specific** overrides when needed
- ❌ **Don't hardcode** models or keys in Python code
- ❌ **Don't commit** `settings.yaml` with real API keys