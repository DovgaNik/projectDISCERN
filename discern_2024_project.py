from transformers import GPT2Tokenizer
import tensorflow as tf
from datetime import datetime
from flask import Flask, jsonify, request

dt_start = datetime.now()
print(f"{dt_start.strftime('%H:%M:%S %m/%d/%y')}: Application started")

model_name = "gpt2"
model_dir = "../the_model_of_the_new_day"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def predict_news_class(probabilities, threshold=0.5):
    fake_probability, true_probability = probabilities[0]
    if fake_probability > threshold:
        return "Fake"
    else:
        return "True"

model = tf.keras.models.load_model(model_dir)

dt_init = datetime.now()
print(f"{dt_init.strftime('%H:%M:%S %m/%d/%y')}: Initialization finished")
print(f"{(dt_init - dt_start).seconds} seconds elapsed")

# while True:
#     dt_startAI = datetime.now()
#     inp = input(f"{dt_startAI.strftime('%H:%M:%S %m/%d/%y')}: Input article: ")
#     inputs = tokenizer(inp, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
#     output = model(inputs)
#     dt_finishAI = datetime.now()
#     print(f"{dt_finishAI.strftime('%H:%M:%S %m/%d/%y')}: {predict_news_class(output)}")
#     print(f"{(dt_finishAI - dt_startAI).seconds} seconds elapsed")

app = Flask(__name__)  # Instantiate your chosen framework

@app.route("/predict" ,methods=['POST', 'GET'])
def predict():
    try:
        data = request.get_json()  # Access JSON data sent in the POST request
        article = data['article']
        inputs = tokenizer(article, return_tensors='tf', max_length=512, truncation=True, padding='max_length')
        output = model(inputs)
        prediction = predict_news_class(output)
        return jsonify({'prediction': prediction})


    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000)  #Specify  host and port