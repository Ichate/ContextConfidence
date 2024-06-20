from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the model
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        data = request.get_json()
        question = data.get('question')
        context = data.get('context')
        
        if not question or not context:
            raise ValueError("Both question and context must be provided.")
        
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
        score = result['score']

        return jsonify({'answer': answer, 'score': score})
    except Exception as e:
        return jsonify({'error': str(e)})

#made by ichate UwU

if __name__ == '__main__':
    app.run(debug=True)
