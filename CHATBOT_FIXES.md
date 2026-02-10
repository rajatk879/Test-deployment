# Chatbot Integration Fixes

## Issues Found and Fixed

### 1. ✅ Fixed Import Path Issues
**Problem**: All files in `Text2SQL_V2/` were using absolute imports like `from Text2SQL_V2.core.db_builder import ...` which don't work when `Text2SQL_V2` is added to `sys.path`.

**Files Fixed**:
- `Text2SQL_V2/chatbot_api.py` - Changed all `Text2SQL_V2.*` imports to relative imports
- `Text2SQL_V2/agents/text2sql_agent.py` - Fixed `from Text2SQL_V2.utils.llm_factory` → `from utils.llm_factory`
- `Text2SQL_V2/agents/summarizer_agent.py` - Fixed import
- `Text2SQL_V2/core/db_builder.py` - Fixed import
- `Text2SQL_V2/utils/llm_factory.py` - Fixed `from Text2SQL_V2.config` → `from config`
- `Text2SQL_V2/summary_generator.py` - Fixed imports

### 2. ✅ Fixed Frontend Bug
**Problem**: In `Chatbot.tsx`, the code was sending `{ question: input }` but `input` was already cleared to empty string.

**Fix**: Changed to `{ question: question }` to use the saved value.

### 3. ⚠️ Database Needs Rebuild
**Problem**: The database contains old tables from previous schema (`recipe_time`, `recipe_time_weekly`, `flour_demand`, `recipe_allocation`, etc.) that don't exist in the current MC4 schema.

**Solution**: Rebuild the database by running:
```bash
cd backend
python setup_chatbot_db.py
```

This will rebuild the database with only the current MC4 tables.

## Testing the Bot

1. **Rebuild the database**:
   ```bash
   cd backend
   python setup_chatbot_db.py
   ```

2. **Start the backend server**:
   ```bash
   cd backend
   source venv/bin/activate  # if using virtual environment
   python fastapi_server.py
   ```

3. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

4. **Test the chatbot**:
   - Open the frontend in browser
   - Click the chatbot icon
   - Try a query like: "What is the total forecast for SKU001?"

## Required Environment Variables

Make sure you have the following environment variables set (in `.env` file or system environment):

```bash
# For Google Gemini (default)
GOOGLE_API_KEY=your_google_api_key_here

# OR for OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

## Current Schema Tables

The chatbot now supports these 17 tables:
- sku_forecast
- sku_master
- recipe_master
- recipe_eligibility
- mill_master
- mill_load
- mill_load_weekly
- mill_load_monthly
- mill_load_yearly
- mill_capacity
- mill_recipe_schedule
- mill_recipe_rates
- bulk_flour_demand
- recipe_demand
- recipe_mix
- raw_material
- time_dimension

## Troubleshooting

If the bot still doesn't work:

1. **Check backend logs** for import errors or API key issues
2. **Verify database exists**: `ls -la Text2SQL_V2/chatbot.db`
3. **Check API keys**: Make sure `GOOGLE_API_KEY` or `OPENAI_API_KEY` is set
4. **Check CORS**: Backend should allow all origins (already configured)
5. **Check network**: Frontend should be able to reach `http://127.0.0.1:8000`

## Next Steps

1. Rebuild the database with the updated schema
2. Test with a simple query
3. Check backend logs for any remaining errors
4. Verify API keys are configured correctly
