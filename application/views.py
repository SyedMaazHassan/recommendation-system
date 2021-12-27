from django.shortcuts import render, redirect
from .models import *
from django.contrib import messages
from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.conf import settings
from django.contrib.auth.decorators import login_required
import os
# main page function

def calculate_response(answers_array):
    base_dir = settings.BASE_DIR
    csv_path = os.path.join(base_dir, "static", "society_root.csv")

    # package for data analysis
    import pandas as pd

    #import Decision Tree Classifier model
    from sklearn .tree import DecisionTreeClassifier

    # to ignore warnings
    import warnings
    warnings.filterwarnings('ignore')

    #--------reading the .csv file as input------------#
    society_data = pd.read_csv(csv_path)

    #Spliting the input and output from the dataset

    #input columns
    X = society_data.drop(columns=['Suggestion'])

    #output column
    Y = society_data['Suggestion']

    # Printing the dataset shape
    # print ("Dataset Length: ", len(society_data))
    # print ("Dataset Shape: ", society_data.shape)

    # Printing the dataset obseravtions
    # print ("Dataset: ",society_data.head())

    #------------------Training and testing with Decision Tree----------------#
    model = DecisionTreeClassifier()
    model.fit(X, Y)

    #There are 12 input columns and one output column in the dataset
    #'0' means 'No'
    #'1' means 'Yes'
    #For predictions
    #model.predict([[Q1_ans, Q2_ans, Q3_ans, Q4_ans, Q5_ans, Q6_ans, Q7_ans, Q8_ans, Q9_ans, Q10_ans, Q11_ans, Q12_ans]])

    predictions = model.predict([answers_array])
    return predictions.tolist()[0].split(",")



def submit_response(request):
    
    output = {
        'status': False,
        'message': None
    }

    if request.method == "GET":
        response = request.GET.get("response")
        if response:
            response = json.loads(response)

            if request.user.is_authenticated:
                new_response = UserResponse(user = request.user)
            else:
                new_response = UserResponse()
          
            new_response.save()

            answers_array = []
            
            for i in response:
                question_id = i['question']
                answer_id = i['answer']

                question = Question.objects.get(id = question_id)
                answer = Answer.objects.get(id = answer_id)

                new_response.question_answers.create(
                    question = question,
                    answer = answer
                )

                answers_array.append(answer.boolean)

            # new_response.save()
            # print(answers_array)
            model_output = calculate_response(answers_array)
            output['response'] = model_output

            output['status'] = True
        return JsonResponse(output)

def index(request):
    # return render(request, "signin.html")
    if request.user.is_authenticated:
        return redirect("main")

    return render(request, 'index.html')



def main(request):
    all_questions = Question.objects.all()
    context = {
        'all_questions': all_questions,
        'total_questions': all_questions.count()
    }
    return render(request, 'main.html', context)


# function for signup

def signup(request):
    if request.user.is_authenticated:
        return redirect("index")
        
    if request.method == "POST":
        name = request.POST['name']
        l_name = request.POST['l_name']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        context = {
            "name":name,
            "l_name":l_name,
            "email":email,
            "pass1":pass1,
            "pass2":pass2,
        }
        if pass1==pass2:
            if User.objects.filter(username=email).exists():
                print("Email already taken")
                messages.info(request, "Entered email already in use!")
                context['border'] = "email" 
                return render(request, "signup.html", context)

            user = User.objects.create_user(username=email, first_name=name, password=pass1, last_name=l_name)
            user.save()
            messages.success(request, "Your account has been created!")
            return redirect("login")
        else:
            messages.info(request, "Your pasword doesn't match!")
            context['border'] = "password"
            return render(request, "signup.html", context)


    
    return render(request, "signup.html")


# function for login

def login(request):
    if request.user.is_authenticated:
        return redirect("index")

    if request.method == "POST":
        email = request.POST['email']
        password = request.POST['password']
        context = {
            'email': email,
            'password': password
        }
        user = auth.authenticate(username=email, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect("index")
        else:
            messages.error(request, "Incorrect login details!")
            return render(request, "login.html", context)
            # return redirect("login")
    else:
        return render(request, "login.html")


# function for logout

def logout(request):
    auth.logout(request)
    return redirect("index")



def department(request):
    all_questions = Dept_Question.objects.all()
    context = {
        'all_questions': all_questions,
        'total_questions': all_questions.count()
    }
    return render(request, 'department.html', context)

def department_result(request):
    return render(request, 'department_result.html')



def dept_calculate_response(answers_array):
    base_dir = settings.BASE_DIR
    csv_path = os.path.join(base_dir, "static", "Department_root.csv")
  
    # package for data analysis
    import pandas as pd

    #import XGBClassifier model
    from xgboost import XGBClassifier

    # to ignore warnings
    import warnings
    warnings.filterwarnings('ignore')

    #--------reading the .csv file as input------------#
    department_data = pd.read_csv(csv_path)

    #Spliting the input and output from the dataset

    #input columns
    X = department_data.drop(columns=['Suggestion'])

    #output column
    Y = department_data['Suggestion']

    #input columns
    X = department_data.drop(columns=['Suggestion'])

    #output column
    Y = department_data['Suggestion']

    #------------------Training and testing with XGBClassifier----------------#
    model = XGBClassifier()
    model.fit(X, Y)

    #------------------Getting prediction----------------#
    df = ([answers_array])
    #--------------------------Adding Headers-----------------------#
    X1 = pd.DataFrame(df,columns=['Communication skills', 'Graphic/web design', 'Problem solving', 'Artistic ability', 'Acting', 'Good with numbers', 'Logical thinking', 'Maths', 'Interest in fashion trends', 'Singing', 'Creative', 'Computer curiousity', 'Like electronic circuits', 'Good in color choice', 'Film-making', 'Leadership skills', 'Programming', 'Fixing things', 'Sewing', 'Variety in work', 'Career in business', 'Imaginative', 'Like electrical circuits', 'Interest in clothing', 'Creative writer', 'Team player', 'Analytical/critical thinker', 'Photography'])

    predictions = model.predict(X1)
    return predictions.tolist()[0].split(",")


def dept_submit_response(request):
    
    output = {
        'status': False,
        'message': None
    }

    if request.method == "GET":
            response = request.GET.get("response")
            if response:
                response = json.loads(response)

                if request.user.is_authenticated:
                    new_response = Dept_UserResponse(user = request.user)
                else:
                    new_response = Dept_UserResponse()

                new_response.save()

                answers_array = []
                
                for i in response:
                    question_id = i['question']
                    answer_id = i['answer']

                    question = Dept_Question.objects.get(id = question_id)
                    answer = Dept_Answer.objects.get(id = answer_id)

                    new_response.question_answers.create(
                        question = question,
                        answer = answer
                    )

                    answers_array.append(answer.boolean)

                new_response.save()
                print(answers_array)
                model_output = dept_calculate_response(answers_array)
                output['response'] = model_output

                output['status'] = True
                return JsonResponse(output)

                return render(request,"department_result.html",{'model_output':model_output})

                