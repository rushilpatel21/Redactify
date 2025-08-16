@echo off
echo Testing TRUE MCP Implementation...
echo.

echo Test 1: Medical Document (should take 10-20 seconds)
curl -X POST http://localhost:8000/anonymize_llm ^
  -H "Content-Type: application/json" ^
  -d "{\"user_request\": \"Anonymize this medical record completely\", \"text\": \"Patient John Smith was treated by Dr. Sarah Johnson at Memorial Hospital. Contact: john@email.com\"}"

echo.
echo.
echo Test 2: Business Email (should choose different strategy)
curl -X POST http://localhost:8000/anonymize_llm ^
  -H "Content-Type: application/json" ^
  -d "{\"user_request\": \"Partially mask this business email\", \"text\": \"Hi Jane Doe, please contact Microsoft Corp. Best regards, Bob Wilson\"}"

echo.
echo.
echo Done! Check the response for LLM reasoning and tool selection.
pause