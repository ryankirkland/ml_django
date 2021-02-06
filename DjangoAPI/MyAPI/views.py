from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
from .serializers import ApprovalSerializer
from .models import Approval
from keras.models import load_model
import numpy as np
import pandas as pd
from pickle import load

# Create your views here.
class ApprovalsView(viewsets.ModelViewSet):
    queryset = Approval.objects.all()
    serializer_class = ApprovalSerializer

@api_view(['POST'])
def approve_reject(request):
    try:
        mdl = load_model('/Users/ryankirkland/Desktop/projects/ml-django/loan_model')
        mydata = request.data
        unit = np.array(mydata.values())
        unit = unit.reshape(1,-1)
        scaler = load((open('/Users/ryankirkland/Desktop/projects/ml-django/scaler.pkl', 'rb')))
        X = scaler.transform(unit)
        y_pred = mdl.predict(X)
        y_pred = (y_pred>0.5)
        newdf = pd.DataFrame(y_pred, columns=['Status']).replace({True: 'Approved', False: 'Rejected'})
        return JsonResponse('Your Statis is {}'.format(newdf), safe=False)
    except ValueError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)