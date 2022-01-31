import PIL.Image

from ImgPred.util import *
from rest_framework.views import APIView



class Main(APIView):
    def post(self, request):
        input_file = InputFile(request.POST, request.FILES)
        input_url = InputURL(request.POST, request.data)
        if input_file.is_valid():
            filepath = save_input_file(request.FILES['file'])
            labels = predict_image(filepath)
            return render(request, "output.html", {'class1': list(labels.keys())[0], 'prob1': list(labels.values())[0],
                                                   'class2': list(labels.keys())[1], 'prob2': list(labels.values())[1],
                                                   'class3': list(labels.keys())[2], 'prob3': list(labels.values())[2]})
        elif input_url.is_valid():
            filepath = save_input_url(request.data['url'])
            labels = predict_image(filepath)
            return render(request, "output.html", {'class1': list(labels.keys())[0], 'prob1': list(labels.values())[0],
                                                   'class2': list(labels.keys())[1], 'prob2': list(labels.values())[1],
                                                   'class3': list(labels.keys())[2], 'prob3': list(labels.values())[2]})
        else:
            return HttpResponse("Bad Request!")

    def get(self, request):
        input_file = InputFile()
        input_url = InputURL()
        return render(request, "input.html", {'form_file': input_file, 'form_url': input_url})
