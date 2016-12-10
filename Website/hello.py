from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/markov')
def markov():
    return render_template('markov.html')

@app.route('/head_to_head')
def head_to_head():
    return render_template('head_to_head.html')

@app.route('/winning')
def winning():
    return render_template('winning.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/create')
def create():
    return render_template('create.html')


if __name__ == "__main__":
    app.run(debug=True)

