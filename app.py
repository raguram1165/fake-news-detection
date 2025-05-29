from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and vectorizer
with open('model_fakenews.pickle', 'rb') as model_file:
    pac = pickle.load(model_file)

with open('tfid.pickle', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/newscheck')
def newscheck():
    # Get user input from the request
    news_text = request.args.get('news')
    if not news_text:
        return jsonify(result="EMPTY")

    # Transform the input and make predictions
    input_data = [news_text.strip()]
    tfidf_test = tfidf_vectorizer.transform(input_data)
    prediction = pac.predict(tfidf_test)
    
    # Return the prediction result
    return jsonify(result=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
