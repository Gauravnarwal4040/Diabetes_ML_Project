from flask import Flask,render_template,request
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

standard_scaler = pickle.load(open("C:/Users/gaura/OneDrive/Desktop/My_Project/ML Project/Diabetes/Models/standard_scaler.pkl", 'rb'))
model = pickle.load(open("C:/Users/gaura/OneDrive/Desktop/My_Project/ML Project/Diabetes/Models/modelForPridiction.pkl", "rb"))

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route("/predictdata",methods = ['POST',"GET"])
def predict_datapoint():
    if request.method == "POST":
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))



        new_data = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        final_predict = model.predict(new_data)


        if final_predict[0] == 1:
            result = "Diabetic"
        
        else:
            result = "Non Diabetic"

        return render_template("single_prediction.html",result = result)
    
    else:
        return render_template("prediction_page.html")





if __name__ == "__main__":
    app.run(host = "0.0.0.0")