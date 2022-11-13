from django.shortcuts import render,redirect
from .forms import EmployeeForm
from .models import Employee


def homepage(request):
 emp=Employee.objects.all()
 return render(request,'index.html',{'emp':emp})

def insert(request,id=0):
    if request.method == "GET":
        if id==0:
            form = EmployeeForm()
        else:
            emp=Employee.objects.get(id=id)
            form = EmployeeForm(instance=emp)
            return render(request,"insert.html",{'form':form})
    else:
        if id==0:
            form = EmployeeForm(request.POST)
        else:
            emp=Employee.objects.get(id=id)
            form = EmployeeForm(request.POST,instance=emp)
    if form.is_valid():
        form.save()
        return redirect('/') 

def delete(request, id):
 emp = Employee.objects.get(id=id)
 emp.delete()
 return redirect('/')
