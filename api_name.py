from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn.functional as F
from Model_class import Model

# load model and dictionaries
idx_to_chr = {0: ' ', 1: '-', 2: 'a', 3: 'b', 4: 'c', 5: 'd', 6: 'e', 7: 'f', 8: 'g', 9: 'h', 10: 'i', 11: 'j', 12: 'k', 13: 'l', 14: 'm', 15: 'n', 16: 'o', 17: 'p', 18: 'q', 19: 'r', 20: 's', 21: 't', 22: 'u', 23: 'v', 24: 'w', 25: 'x', 26: 'y', 27: 'z', 28: '<E>'}
chr_to_idx = {' ': 0, '-': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6, 'f': 7, 'g': 8, 'h': 9, 'i': 10, 'j': 11, 'k': 12, 'l': 13, 'm': 14, 'n': 15, 'o': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'u': 22, 'v': 23, 'w': 24, 'x': 25, 'y': 26, 'z': 27, '<E>': 28}   

model = Model()
model.load_state_dict(torch.load('name_model_lstm.pt'))
model.eval()

# create flask app
app = Flask(__name__)

@app.route('/')
def temp():
    return render_template('template.html')

@app.route('/',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        info = request.form['search']
        return redirect(url_for('predict',values=info))

@app.route('/predict/<values>')
def predict(values):
    seed = list(values.lower())
    idx1 = chr_to_idx[seed[0]]
    idx2 = chr_to_idx[seed[1]]
    seed = torch.tensor([idx1, idx2], dtype=torch.long)
    seed = seed.unsqueeze(0)
    
    names = []
    for i in range(5):
        output = ['<S>']
        while output[-1] != '<E>':
            logits = model(seed)
            prob = F.softmax(logits.data, dim=1)
            predicted = torch.multinomial(prob, 1).squeeze(1)
            chr_predicted = idx_to_chr[predicted.item()]
            output.append(chr_predicted)
            seed = torch.cat((seed[:, -1], predicted)).unsqueeze(0)
        final_output = [idx_to_chr[idx1], idx_to_chr[idx2]] + output[1:-1]
        output_name = ''.join(final_output)
        names.append(output_name)
    generated_name = '\n'.join(names)
    return render_template('template.html', generated_name=generated_name)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
