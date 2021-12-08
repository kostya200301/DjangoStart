from django.http import HttpResponse
from django.shortcuts import render
from datetime import datetime, date
from utils.pred_back import Get_Pred
def home(request):
    return render(request, 'home.html')#, {'hello': 'Hello Alex !'}
def stock1(request):
    current_date = datetime.now().date()
    return render(request, 'stock1.html', {'curr_date': str(current_date)})
def stock2(request):
    current_date = datetime.now().date()
    return render(request, 'stock2.html', {'curr_date': str(current_date)})
def stock3(request):
    current_date = datetime.now().date()
    return render(request, 'stock3.html', {'curr_date': str(current_date)})
def stock4(request):
    current_date = datetime.now().date()
    return render(request, 'stock4.html', {'curr_date': str(current_date)})
def stock5(request):
    current_date = datetime.now().date()
    return render(request, 'stock5.html', {'curr_date': str(current_date)})
def Preds(request):
    date1 = list(map(int, str(request.GET['start']).split('-')))
    date2 = list(map(int, str(request.GET['end']).split('-')))
    predate = list(map(int, str(request.GET['predz']).split('-')))
    days_diff = (date(predate[0], predate[1], predate[-1]) - date(date2[0], date2[1], date2[-1])).days
    compn = request.GET['company_name']
    return render(request, 'prediction.html', {'stockname': request.GET['stockfull'], 'predz': '/'.join(list(map(str, predate[::-1]))), 'finalpred': Get_Pred(compn.upper(), date1, days_diff)})

