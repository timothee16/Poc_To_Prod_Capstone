from flask import Flask, request, render_template
import run


app = Flask(__name__)   

@app.route("/", methods=['POST', 'GET'])
def index():

    if request.method == "POST":
        question = request.form["question"]
        print(question)
        path = r"C:\Users\timot\EPF\From_POC_to_PROD\rendu_projet\Capstone\poc-to-prod-capstone\poc-to-prod-capstone\train\data\artefacts" 
        prediction_object = run.TextPredictionModel.from_artefacts(path)
        predict = prediction_object.predict([question], top_k=1)
        print("Top_k", predict)
        return (str(predict))
    return render_template("form.html")

if __name__ == '__main__':
    app.run()