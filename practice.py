from openai import OpenAI
import json
from weather import get_weather
from stock import get_stock_price
from calculator import simple_calculator
from schema import tools
from rag_files.rag import search_pdf

client = OpenAI()

conversation_history = []
MAX_HISTORY_ITEMS = 50


tool_map = {
    "get_weather": get_weather,
    "simple_calculator": simple_calculator,
    "get_stock_price": get_stock_price,
    "search_pdf": search_pdf,
}


def trim_history():
    global conversation_history
    if len(conversation_history) > MAX_HISTORY_ITEMS:
        conversation_history = conversation_history[-MAX_HISTORY_ITEMS:]

def run_agent(user_input: str):

    global conversation_history

    conversation_history.append({
        "role": "user",
        "content": user_input,
    })
    trim_history()

    response = client.responses.create(
        model="gpt-5.5",
        tools=tools,
        input=conversation_history,
    )

    conversation_history += response.output

    for item in response.output:
        print(item.type, getattr(item, "name", None))

        if item.type == "function_call":
            function = tool_map.get(item.name)

            if function is None:
                raise ValueError(f"Unknown tool: {item.name}")

            args = json.loads(item.arguments)

            try:
                result = function(**args)
            except Exception as e:
                result = {
                    "error": str(e),
                    "tool": item.name,
                }

            conversation_history.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(result, ensure_ascii=False),
            })

    stream = client.responses.create(
        model="gpt-5.5",
        instructions=(
            "请根据上下文和工具返回结果，用简洁中文回答用户。"
            "不要输出原始 JSON。"
            "如果需要查询 PDF、天气、股票或计算，请优先使用工具结果。"
        ),
        tools=tools,
        input=conversation_history,
        stream = True
    )

    answer_text = ""

    print("\nAI:", end=" ")

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta,end = "",flush = True)
            answer_text+= event.delta

    print()

    conversation_history.append({
        "role": "assistant",
        "content": answer_text,
    })

    trim_history()

    return answer_text

def main():
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Bye!")
            break

        run_agent(user_input)



if __name__ == "__main__":
    main()