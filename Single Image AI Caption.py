import base64,requests
from config import HF_API_KEY
Api_Url="https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}
Models=["zai-org/GLM-4.5V",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
    "google/gemma-3-27b-it"]
def data_url(b:bytes)->str:
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"
def extract_err(r:requests.Response)->str:
    try:
        return r.json()["error"]["message"]
    except:
        return r.text
def box(model:str,image:bytes,title:str,lines:list[str],icon:str)->str:
    w=max(30,len(title)+4,*(len(x) for x in lines))
    print("\n" + "┏" + "━" * (w + 2) + "┓")
    print(f"┃ {icon} {title.ljust(w - 2)} ┃")
    print("┣" + "━" * (w + 2) + "┫")
    for line in lines:
        print(f"┃ {line.ljust(w - 2)} ┃")
    print("┗" + "━" * (w + 2) + "┛")
    return f"```{model}```"
def caption_single_image():
    image_source=input("Enter the path of the image: ")
    try:
        with open(image_source,"rb") as f:
            img=f.read()
    except:
        print("Error: Invalid image path")
        return
    base={
        "messages":[{
            "role":"user",
            "content":[{
             "type":"text","text":"Describe this image: "},
                {"type":"image_url","image_url":{"url":data_url(img)}}]

        }]
    }
    last = None
    for model in Vision_models:
        payload = {
            "model": model,
            "inputs": base,
            "parameters": {
                "max_new_tokens": 220,
                "temperature": 0.3,
                "top_p": 0.9,
                "do_sample": True,
                "stream": False
            }
        }
        try:
            data = query_hf_api(payload)
            last = _extract_text(data)
            print(f"```{model}```\n{last}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    return last
def main():
    print_menu()
    while True:
        choice = input("Enter your choice: ")
        if choice == "1":
            get_basic_caption()
        elif choice == "2":
            break
        else:
            print("Invalid choice. Please try again.")
if __name__ == "__main__":
    main()
    
