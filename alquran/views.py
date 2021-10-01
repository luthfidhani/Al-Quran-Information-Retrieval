from django.shortcuts import render, reverse
from django.http import HttpResponseRedirect
from .IrProcessing import IR

def index(request):
    return render(request, 'alquran/index.html')

def analyze(request):
    text = request.POST['query-alquran']
    if text:
        ir = IR.Execute(text).tolist()
        return render(request, 'alquran/index.html', {'query':text, 'datas': ir})

    return HttpResponseRedirect(reverse('index'))