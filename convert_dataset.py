import json

with open("dataset.json", "r") as f:
    data = json.load(f)

out = []
for example in data:
    messages = example.get("messages", [])
    for i in range(len(messages) - 1):
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            out.append({
                "instruction": messages[i]["content"],
                "output": messages[i+1]["content"]
            })

with open("hf_train.jsonl", "w") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Converted {len(out)} examples to hf_train.jsonl")